#!/usr/bin/env python3
"""
Generate experiment circuit sets: one L3-transpiled original + N remapped variants.

Each circuit gets its own artifacts subdirectory, ready for `python -m quantum.cli`.

Usage:
    python circuits/generate_experiments.py --algorithm grover --remap-count 2
    python circuits/generate_experiments.py --algorithm simon --secret 10 --remap-count 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qiskit import qpy

from common import transpile_and_save
from grover_3_qubits import grover_3q_circuit
from simon_3_qubits import simon_3q_circuit
from remap_qpy_layouts import (
    get_active_qubits,
    build_compact_circuit,
    required_logical_edges,
    read_backend_topology,
    build_neighbors,
    propose_backend_driven_subsets,
    feasible_assignments_for_subset,
    score_mapping,
    remap_compact_circuit,
    ensure_backend,
)
from quantum.artifacts import save_circuit_with_diagram


def generate_experiment_set(
    algorithm: str,
    remap_count: int = 2,
    secret: str = "10",
    marked: int = 7,
    output_root: Path = Path("artifacts"),
):
    """Generate original L3 circuit + remapped variants."""
    # 1) Build source circuit
    if algorithm == "grover":
        source = grover_3q_circuit(marked_states=[marked])
        prefix = "grover3"
    elif algorithm == "simon":
        source = simon_3q_circuit(secret_string=secret)
        prefix = "simon3"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # 2) Transpile and save original
    original_dir = output_root / f"{prefix}_original"
    isa, qpy_path, png_path = transpile_and_save(source, original_dir, f"{prefix}_l3")
    print(f"Original: {qpy_path}")

    if remap_count <= 0:
        return

    # 3) Remap
    active = get_active_qubits(isa)
    compact = build_compact_circuit(isa, active)
    req_edges = required_logical_edges(compact)

    backend = ensure_backend("quantum_config.json", None)
    backend_name = getattr(backend, "name", None)
    backend_name = backend_name() if callable(backend_name) else backend_name

    op_qubits, op_edges, props = read_backend_topology(backend)
    neighbors = build_neighbors(op_qubits, op_edges)
    subsets = propose_backend_driven_subsets(
        op_qubits, neighbors, props, backend, compact.num_qubits, max_candidates=1000,
    )
    if not subsets:
        raise RuntimeError("No connected physical subsets found.")

    # Score and deduplicate mappings
    candidates = []
    seen = set()
    for subset in subsets:
        for m in feasible_assignments_for_subset(subset, compact.num_qubits, req_edges, op_edges):
            key = tuple(m)
            if key in seen:
                continue
            seen.add(key)
            sc, details = score_mapping(m, req_edges, backend, props)
            candidates.append((m, sc, details))

    if not candidates:
        raise RuntimeError("No feasible mapping found.")

    # Deduplicate by qubit set, sort by score
    candidates.sort(key=lambda x: x[1])
    chosen = []
    seen_sets = set()
    for m, sc, details in candidates:
        key = tuple(sorted(m))
        if key not in seen_sets:
            seen_sets.add(key)
            chosen.append((m, sc, details))
        if len(chosen) >= remap_count:
            break

    # 4) Save each remap to its own directory
    for idx, (mapping, sc, details) in enumerate(chosen, start=1):
        remap_dir = output_root / f"{prefix}_remap_{idx:02d}"
        remapped = remap_compact_circuit(compact, mapping)
        remap_name = f"{prefix}_l3_remap_{idx:02d}_{backend_name}"
        qpy_path, png_path = save_circuit_with_diagram(remapped, remap_dir, remap_name)

        # Save mapping metadata
        meta = {"mapping": mapping, "score": sc, "score_details": details, "backend": backend_name}
        (remap_dir / "mapping_info.json").write_text(json.dumps(meta, indent=2))
        print(f"Remap {idx}: {qpy_path}  mapping={mapping}  score={sc:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment circuit sets")
    parser.add_argument("--algorithm", "-a", required=True, choices=["grover", "simon"])
    parser.add_argument("--remap-count", "-n", type=int, default=2)
    parser.add_argument("--secret", default="10", help="Simon secret string")
    parser.add_argument("--marked", type=int, default=7, help="Grover marked state")
    parser.add_argument("--output", "-o", default="artifacts", help="Output root directory")
    args = parser.parse_args()

    generate_experiment_set(
        algorithm=args.algorithm,
        remap_count=args.remap_count,
        secret=args.secret,
        marked=args.marked,
        output_root=Path(args.output),
    )


if __name__ == "__main__":
    main()
