#!/usr/bin/env python3
"""
Generate remapped QPY circuits on valid physical qubit layouts.

Given one input QPY circuit, this script:
1) Compacts active qubits into a logical circuit.
2) Connects to the configured IBM backend.
3) Finds candidate physical mappings that are operational and connected.
4) Exports remapped QPY files plus a summary JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from qiskit import QuantumCircuit, qpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.runtime_ops import QuantumServiceManager


@dataclass
class CandidateMapping:
    logical_to_physical: List[int]
    score: float
    score_details: Dict[str, float]


def load_first_qpy_circuit(qpy_path: Path) -> QuantumCircuit:
    with qpy_path.open("rb") as f:
        circuits = list(qpy.load(f))
    if not circuits:
        raise ValueError(f"No circuits in QPY: {qpy_path}")
    return circuits[0]


def get_active_qubits(circuit: QuantumCircuit) -> List[int]:
    used = sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits})
    if not used:
        raise ValueError("Input circuit has no active qubits.")
    return used


def build_compact_circuit(circuit: QuantumCircuit, active_qubits: Sequence[int]) -> QuantumCircuit:
    old_to_new = {old: i for i, old in enumerate(active_qubits)}
    compact = QuantumCircuit(len(active_qubits), circuit.num_clbits, name=f"{circuit.name or 'circuit'}_compact")

    for inst in circuit.data:
        q_old = [circuit.find_bit(q).index for q in inst.qubits]
        c_idx = [circuit.find_bit(c).index for c in inst.clbits]
        q_new = [old_to_new[q] for q in q_old]

        compact.append(
            inst.operation,
            [compact.qubits[i] for i in q_new],
            [compact.clbits[i] for i in c_idx],
        )
    return compact


def required_logical_edges(circuit: QuantumCircuit) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for inst in circuit.data:
        # Only 2-qubit quantum gates impose direct coupling requirements.
        # Multi-qubit directives like barrier should not create topology constraints.
        if inst.operation.name in {"measure", "barrier"}:
            continue
        q_idx = [circuit.find_bit(q).index for q in inst.qubits]
        if len(q_idx) != 2:
            continue
        a, b = q_idx
        if a == b:
            continue
        edges.add(tuple(sorted((a, b))))
    return edges


def _to_undirected_edges(raw_edges: Iterable[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for a, b in raw_edges:
        if a == b:
            continue
        out.add(tuple(sorted((int(a), int(b)))))
    return out


def read_backend_topology(backend) -> Tuple[Set[int], Set[Tuple[int, int]], object]:
    num_qubits = int(getattr(backend, "num_qubits"))

    coupling = getattr(backend, "coupling_map", None)
    if coupling is None:
        target = getattr(backend, "target", None)
        if target is not None and getattr(target, "build_coupling_map", None):
            coupling = target.build_coupling_map()
    if coupling is None:
        raise RuntimeError("Backend has no coupling map; cannot layout mapped circuits safely.")

    edges_raw: List[Tuple[int, int]] = []
    if hasattr(coupling, "get_edges"):
        edges_raw = list(coupling.get_edges())
    else:
        edges_raw = list(coupling)

    props = backend.properties()
    faulty_qubits: Set[int] = set()
    if props is not None and hasattr(props, "faulty_qubits"):
        try:
            faulty_qubits = set(int(q) for q in props.faulty_qubits())
        except Exception:
            faulty_qubits = set()

    operational_qubits = set(range(num_qubits)) - faulty_qubits

    edges = _to_undirected_edges(edges_raw)
    operational_edges = {
        (a, b) for (a, b) in edges if a in operational_qubits and b in operational_qubits
    }
    return operational_qubits, operational_edges, props


def build_neighbors(
    qubits: Set[int], edges: Set[Tuple[int, int]]
) -> Dict[int, Set[int]]:
    neighbors: Dict[int, Set[int]] = {q: set() for q in qubits}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    return neighbors


def is_connected_subset(subset: Set[int], neighbors: Dict[int, Set[int]]) -> bool:
    if not subset:
        return False
    root = next(iter(subset))
    seen = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nb in neighbors[cur]:
            if nb in subset and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(subset)


def enumerate_connected_subsets(
    qubits: Set[int],
    neighbors: Dict[int, Set[int]],
    subset_size: int,
) -> List[Set[int]]:
    """Enumerate all connected subsets of given size."""
    partial_seen: Set[frozenset[int]] = set()
    full_seen: Set[frozenset[int]] = set()
    stack: List[Set[int]] = [{q} for q in sorted(qubits)]

    while stack:
        subset = stack.pop()
        frozen = frozenset(subset)
        if frozen in partial_seen:
            continue
        partial_seen.add(frozen)

        if len(subset) == subset_size:
            full_seen.add(frozen)
            continue
        if len(subset) > subset_size:
            continue

        frontier = set()
        for q in subset:
            frontier.update(nb for nb in neighbors[q] if nb not in subset)
        for nb in frontier:
            new_subset = set(subset)
            new_subset.add(nb)
            stack.append(new_subset)

    return [set(s) for s in full_seen]


def _min_2q_error(props, a: int, b: int, gate_priority: Sequence[str]) -> Optional[float]:
    best: Optional[float] = None
    for gate in gate_priority:
        for pair in ([a, b], [b, a]):
            ge = _safe_gate_error(props, gate, pair)
            if ge is not None and (best is None or ge < best):
                best = ge
    return best


def propose_backend_driven_subsets(
    qubits: Set[int],
    neighbors: Dict[int, Set[int]],
    props,
    backend,
    subset_size: int,
    max_candidates: int,
) -> List[Set[int]]:
    """
    Build candidate connected subsets guided by backend calibration quality.
    This avoids pure combinational brute-force on all subsets.
    """
    op_names = set(getattr(backend, "operation_names", []) or [])
    twoq_gate_priority = [g for g in ["cz", "ecr", "cx"] if g in op_names]

    def node_cost(q: int) -> float:
        r = _safe_readout_error(props, q)
        if r is not None:
            return r
        return 1.0

    ordered_roots = sorted(qubits, key=node_cost)
    unique_sets: Set[frozenset[int]] = set()

    for root in ordered_roots:
        if len(unique_sets) >= max_candidates:
            break
        subset = {root}
        while len(subset) < subset_size:
            frontier = set()
            for q in subset:
                frontier.update(nb for nb in neighbors[q] if nb not in subset)
            if not frontier:
                break

            best_nb = None
            best_cost = None
            for nb in frontier:
                nb_cost = node_cost(nb)
                edge_costs = []
                for q in subset:
                    if nb in neighbors[q]:
                        v = _min_2q_error(props, q, nb, twoq_gate_priority)
                        if v is not None:
                            edge_costs.append(v)
                avg_edge = (sum(edge_costs) / len(edge_costs)) if edge_costs else 1.0
                cost = 0.7 * avg_edge + 0.3 * nb_cost
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_nb = nb

            if best_nb is None:
                break
            subset.add(best_nb)

        if len(subset) == subset_size and is_connected_subset(subset, neighbors):
            unique_sets.add(frozenset(subset))

    return [set(s) for s in unique_sets]


def feasible_assignments_for_subset(
    subset: Set[int],
    logical_size: int,
    required_edges: Set[Tuple[int, int]],
    physical_edges: Set[Tuple[int, int]],
) -> List[List[int]]:
    out: List[List[int]] = []
    phys = sorted(subset)
    for perm in permutations(phys, logical_size):
        ok = True
        for u, v in required_edges:
            if tuple(sorted((perm[u], perm[v]))) not in physical_edges:
                ok = False
                break
        if ok:
            out.append(list(perm))
    return out


def _safe_readout_error(props, q: int) -> Optional[float]:
    if props is None or not hasattr(props, "readout_error"):
        return None
    try:
        value = props.readout_error(q)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_gate_error(props, gate: str, qubits: Sequence[int]) -> Optional[float]:
    if props is None or not hasattr(props, "gate_error"):
        return None
    try:
        value = props.gate_error(gate, list(qubits))
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def score_mapping(
    mapping: Sequence[int],
    required_edges: Set[Tuple[int, int]],
    backend,
    props,
) -> Tuple[float, Dict[str, float]]:
    op_names = set(getattr(backend, "operation_names", []) or [])

    readouts: List[float] = []
    for q in mapping:
        v = _safe_readout_error(props, q)
        if v is not None:
            readouts.append(v)
    avg_readout = sum(readouts) / len(readouts) if readouts else 0.0

    oneq_gate_priority = [g for g in ["sx", "x", "rz"] if g in op_names]
    oneq_vals: List[float] = []
    for q in mapping:
        vals = []
        for gate in oneq_gate_priority:
            ge = _safe_gate_error(props, gate, [q])
            if ge is not None:
                vals.append(ge)
        if vals:
            oneq_vals.append(min(vals))
    avg_1q = sum(oneq_vals) / len(oneq_vals) if oneq_vals else 0.0

    twoq_gate_priority = [g for g in ["cz", "ecr", "cx"] if g in op_names]
    twoq_vals: List[float] = []
    for u, v in required_edges:
        pu, pv = mapping[u], mapping[v]
        best: Optional[float] = None
        for gate in twoq_gate_priority:
            for pair in ([pu, pv], [pv, pu]):
                ge = _safe_gate_error(props, gate, pair)
                if ge is not None and (best is None or ge < best):
                    best = ge
        if best is not None:
            twoq_vals.append(best)
    avg_2q = sum(twoq_vals) / len(twoq_vals) if twoq_vals else 0.0

    score = 0.5 * avg_2q + 0.3 * avg_readout + 0.2 * avg_1q
    details = {
        "avg_2q_error": avg_2q,
        "avg_readout_error": avg_readout,
        "avg_1q_error": avg_1q,
    }
    return score, details


def remap_compact_circuit(compact: QuantumCircuit, mapping: Sequence[int]) -> QuantumCircuit:
    out_n_qubits = max(mapping) + 1
    remapped = QuantumCircuit(out_n_qubits, compact.num_clbits, name=f"{compact.name}_remap")

    for inst in compact.data:
        q_idx = [compact.find_bit(q).index for q in inst.qubits]
        c_idx = [compact.find_bit(c).index for c in inst.clbits]
        remapped_q = [mapping[i] for i in q_idx]
        remapped.append(
            inst.operation,
            [remapped.qubits[i] for i in remapped_q],
            [remapped.clbits[i] for i in c_idx],
        )

    remapped.metadata = dict(compact.metadata or {})
    remapped.metadata.update(
        {
            "logical_to_physical": {f"l{i}": int(p) for i, p in enumerate(mapping)},
            "logical_qubits": compact.num_qubits,
        }
    )
    return remapped


def ensure_backend(config_file: str, backend_name: Optional[str], allow_fake: bool = False):
    svc = QuantumServiceManager(config_file=config_file)
    if not svc.connect():
        raise RuntimeError(f"Failed to connect to IBM service: {svc.last_connect_error}")

    if backend_name:
        backend = svc.service.backend(backend_name)
    else:
        backend = svc.select_backend()
    if backend is None:
        raise RuntimeError("Failed to select backend.")
    return backend


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate remapped QPY circuits on valid backend physical qubits.",
    )
    parser.add_argument(
        "--input-qpy",
        required=True,
        help="Input QPY file path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for remapped QPYs. Default: sibling folder next to input.",
    )
    parser.add_argument(
        "--config",
        default="quantum_config.json",
        help="Quantum config JSON path (default: quantum_config.json).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional backend override. Default uses config backend.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of remapped QPY files to generate. Use 0 for all unique sets (default: 5).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=1000,
        help="Max backend-driven connected subsets to evaluate (default: 1000).",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Enumerate all connected subsets (slow). Default uses backend-driven search.",
    )
    args = parser.parse_args()

    input_qpy = Path(args.input_qpy).expanduser().resolve()
    if not input_qpy.exists():
        raise FileNotFoundError(f"Input QPY not found: {input_qpy}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = input_qpy.parent / f"{input_qpy.stem}_remapped_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input QPY: {input_qpy}")
    original = load_first_qpy_circuit(input_qpy)
    active = get_active_qubits(original)
    compact = build_compact_circuit(original, active)
    req_edges = required_logical_edges(compact)

    print(f"Original qubits: {original.num_qubits}, active qubits: {active}")
    print(f"Logical compact qubits: {compact.num_qubits}, required edges: {sorted(req_edges)}")

    backend = ensure_backend(args.config, args.backend)
    backend_name = getattr(backend, "name", None)
    backend_name = backend_name() if callable(backend_name) else backend_name
    print(f"Backend: {backend_name}")

    operational_qubits, operational_edges, props = read_backend_topology(backend)
    neighbors = build_neighbors(operational_qubits, operational_edges)
    if args.exhaustive:
        subsets = enumerate_connected_subsets(
            qubits=operational_qubits,
            neighbors=neighbors,
            subset_size=compact.num_qubits,
        )
        print(f"Enumerated connected subsets: {len(subsets)}")
    else:
        subsets = propose_backend_driven_subsets(
            qubits=operational_qubits,
            neighbors=neighbors,
            props=props,
            backend=backend,
            subset_size=compact.num_qubits,
            max_candidates=max(1, args.max_candidates),
        )
        print(f"Backend-driven connected subsets: {len(subsets)}")
    if not subsets:
        raise RuntimeError("No connected physical subsets found.")

    mapping_candidates: List[CandidateMapping] = []
    seen_mappings: Set[Tuple[int, ...]] = set()
    for subset in subsets:
        feasible = feasible_assignments_for_subset(
            subset=subset,
            logical_size=compact.num_qubits,
            required_edges=req_edges,
            physical_edges=operational_edges,
        )
        for m in feasible:
            key = tuple(m)
            if key in seen_mappings:
                continue
            seen_mappings.add(key)
            score, details = score_mapping(m, req_edges, backend, props)
            mapping_candidates.append(
                CandidateMapping(logical_to_physical=m, score=score, score_details=details)
            )

    if not mapping_candidates:
        raise RuntimeError("No feasible mapping found for required interaction edges.")

    mapping_candidates.sort(key=lambda x: x.score)
    unique_by_set: List[CandidateMapping] = []
    seen_sets: Set[Tuple[int, ...]] = set()
    for cand in mapping_candidates:
        key = tuple(sorted(cand.logical_to_physical))
        if key in seen_sets:
            continue
        seen_sets.add(key)
        unique_by_set.append(cand)

    chosen = unique_by_set if args.count <= 0 else unique_by_set[: max(1, args.count)]
    print(
        f"Feasible mappings found: {len(mapping_candidates)}, "
        f"unique qubit sets: {len(unique_by_set)}, exporting: {len(chosen)}"
    )

    summary = {
        "input_qpy": str(input_qpy),
        "backend": backend_name,
        "generated_at": ts,
        "active_qubits_in_input": active,
        "logical_qubit_count": compact.num_qubits,
        "required_logical_edges": sorted([list(e) for e in req_edges]),
        "mappings": [],
    }

    for idx, cand in enumerate(chosen, start=1):
        remapped = remap_compact_circuit(compact, cand.logical_to_physical)
        out_name = f"{input_qpy.stem}_remap_{idx:02d}.qpy"
        out_path = output_dir / out_name
        with out_path.open("wb") as f:
            qpy.dump(remapped, f)

        summary["mappings"].append(
            {
                "rank": idx,
                "qpy_file": out_name,
                "logical_to_physical": cand.logical_to_physical,
                "score": cand.score,
                "score_details": cand.score_details,
            }
        )
        print(f"[{idx}] {out_name} -> {cand.logical_to_physical} (score={cand.score:.6g})")

    summary_path = output_dir / "mapping_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
