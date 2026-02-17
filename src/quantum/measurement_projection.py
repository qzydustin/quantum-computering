from collections import defaultdict
from typing import Dict, List


def project_counts(
    counts: Dict[str, int],
    measured_qubits: List[int],
    measure_map: Dict[int, int],
    allow_fallback: bool = False,
    log_warnings: bool = False,
) -> Dict[str, int]:
    """Project full measurement counts onto a selected measured-qubit order."""
    acc = defaultdict(int)

    missing_qubits = [q for q in measured_qubits if q not in measure_map]
    if missing_qubits:
        if not allow_fallback:
            return {}

        if log_warnings:
            print(f"⚠️  Warning: Some measured qubits not in measure_map: {missing_qubits}")
            print(f"   measure_map keys: {list(measure_map.keys())}")
            print(f"   measured_qubits: {measured_qubits}")

        if not counts:
            return {}

        first_key = next(iter(counts.keys()))
        bitstring_length = len(first_key.replace(" ", ""))
        if bitstring_length == len(measured_qubits):
            clbits_in_order = list(range(bitstring_length))
        else:
            clbits_in_order = list(range(min(bitstring_length, len(measured_qubits))))
    else:
        clbits_in_order = [measure_map[q] for q in measured_qubits]

    if not clbits_in_order:
        return {}

    max_clbit_index = max(clbits_in_order)
    for bitstring, value in (counts or {}).items():
        bits = bitstring.replace(" ", "")
        if len(bits) < (max_clbit_index + 1):
            continue
        try:
            projected_bits = ''.join(bits[::-1][c] for c in clbits_in_order)
            acc[projected_bits] += value
        except (IndexError, KeyError):
            continue

    return dict(acc)
