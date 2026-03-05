"""
Cross-report analysis and verification of problematic segments.

Reads multiple delta-debug report JSONs, finds segments that recur
across runs, builds minimal circuits for the top-k, and optionally
verifies each against real hardware via TVD.
"""

import json
import glob
import os
from collections import defaultdict
from typing import List, Dict, Any

from qiskit import QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping

from .executor import QuantumExecutor
from .metrics import calculate_tvd

_GATE_MAP = get_standard_gate_name_mapping()


def _segment_key(segment: Dict[str, Any]) -> str:
    """Canonical string key for a segment based on its operations."""
    parts = []
    for inst in segment.get("instructions", []):
        q = ",".join(map(str, inst["qubits"]))
        p = ",".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                     for v in inst.get("params", []))
        parts.append(f"{inst['operation']}({q})" if not p else f"{inst['operation']}({q};{p})")
    return " | ".join(parts)


def analyze_reports(data_dir: str) -> List[Dict[str, Any]]:
    """Load delta-debug reports and rank segments by recurrence.

    Returns list sorted by frequency, each with:
        key, count, report_indices, segment
    """
    paths = sorted(set(
        glob.glob(os.path.join(data_dir, "*.json"))
        + glob.glob(os.path.join(data_dir, "*", "*.json"))
    ))
    if not paths:
        print(f"No reports found in {data_dir}/")
        return []

    freq: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "reports": set(), "segment": None}
    )
    for idx, path in enumerate(paths):
        with open(path) as f:
            report = json.load(f)
        seg_by_id = {s["layer_id"]: s for s in report.get("segments_info", [])}
        for seg_id in report.get("problematic_segments", []):
            seg = seg_by_id.get(seg_id)
            if seg is None:
                continue
            key = _segment_key(seg)
            freq[key]["count"] += 1
            freq[key]["reports"].add(idx)
            if freq[key]["segment"] is None:
                freq[key]["segment"] = seg

    print(f"Loaded {len(paths)} reports, found {len(freq)} unique problematic segments")
    results = [
        {"key": k, "count": d["count"], "report_indices": sorted(d["reports"]), "segment": d["segment"]}
        for k, d in freq.items()
    ]
    results.sort(key=lambda x: (-x["count"], -len(x["report_indices"])))
    return results


def build_circuit_from_segment(segment: Dict[str, Any]) -> QuantumCircuit:
    """Build a minimal circuit from a segment's instructions.

    Uses original physical qubit indices so the circuit runs on the
    same backend without remapping.
    """
    all_qubits = set()
    for inst in segment["instructions"]:
        all_qubits.update(inst["qubits"])

    num_qubits = max(all_qubits) + 1
    qc = QuantumCircuit(num_qubits, len(all_qubits))

    for inst in segment["instructions"]:
        gate_cls = _GATE_MAP.get(inst["operation"])
        if gate_cls is None:
            raise ValueError(f"Unknown gate: {inst['operation']}")
        qc.append(gate_cls(*inst.get("params", [])), inst["qubits"])

    for i, q in enumerate(sorted(all_qubits)):
        qc.measure(q, i)
    return qc


def verify_segments(
    executor: QuantumExecutor,
    ranked_segments: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Build circuits from top-k segments and verify via TVD.

    Args:
        executor: Initialized QuantumExecutor
        ranked_segments: Output of analyze_reports()
        top_k: Number of top segments to verify

    Returns:
        List of verification results with tvd_loss per segment.
    """
    results = []
    for entry in ranked_segments[:top_k]:
        seg = entry["segment"]
        label = seg.get("description", entry["key"])
        print(f"\nVerifying: {label}  (appeared {entry['count']}x)")

        try:
            circuit = build_circuit_from_segment(seg)
        except ValueError as e:
            print(f"  Skipped: {e}")
            continue

        noisy = executor.run_circuit(circuit, execution_type="noisy_simulator")
        if not noisy.get("success"):
            print(f"  Noisy sim failed: {noisy.get('error')}")
            continue

        real = executor.run_circuit(circuit, execution_type="real_device")
        if not real.get("success"):
            real = executor.run_circuit(circuit, execution_type="real_device")
        if not real.get("success"):
            print(f"  Real device failed: {real.get('error')}")
            continue

        tvd, _ = calculate_tvd(noisy["counts"], real["counts"])
        print(f"  TVD: {tvd:.6f}  (job: {real.get('job_id')})")

        results.append({
            "key": entry["key"],
            "description": label,
            "count": entry["count"],
            "report_indices": entry["report_indices"],
            "tvd_loss": tvd,
            "job_id": real.get("job_id"),
        })

    return results
