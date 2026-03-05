from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence, Tuple, Callable
from datetime import datetime
import json

from qiskit import QuantumCircuit
from .metrics import calculate_tvd


class QuantumDeltaDebugger:
    """
    Quantum Circuit Delta Debugger (based on DDMin)

    - Targets `QuantumCircuit` compiled to backend ISA
    - Measures "loss" using Total Variation Distance between baseline and test distributions
    - Uses DDMin to find the minimal set of circuit segments causing the maximum loss
    """

    TWO_QUBIT_GATES = frozenset([
        "cx", "cy", "cz", "ch", "swap", "iswap", "dcx", "ecr",
        "rzz", "rxx", "ryy", "rzx",
        "crx", "cry", "crz", "cp", "cu", "cu1", "cu3",
        "mcx", "mcy", "mcz", "ccx", "ccy", "ccz",
    ])
    SKIP_OPS = frozenset(["measure", "barrier", "delay", "reset"])
    MAX_LAYER_SIZE = 5

    def __init__(
        self,
        executor,
        tolerance: float = 0.01,
        test_mode: bool = False,
        max_granularity: int = 16,
    ) -> None:
        self.executor = executor
        self.tolerance = float(tolerance)
        self.test_mode: bool = bool(test_mode)
        self.max_granularity = int(max_granularity)

        if self.test_mode:
            self.test_execution_type = "noisy_simulator"
            self.baseline_execution_type = "ideal_simulator"
        else:
            self.test_execution_type = "real_device"
            self.baseline_execution_type = "noisy_simulator"

        self.original_circuit: Optional[QuantumCircuit] = None
        self.segments: List[Dict[str, Any]] = []
        self.measured_qubits_list: List[int] = []
        self.measure_map: Dict[int, int] = {}
        self.test_count: int = 0
        self.ddmin_log: List[Dict[str, Any]] = []
        self._on_step: Optional[Callable[[Dict[str, Any]], None]] = None

    # ---------- segmentation ----------
    def extract_circuit_segments(self, circuit: QuantumCircuit) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        current_layer: List[Dict[str, Any]] = []

        def flush():
            if not current_layer:
                return
            seg = {
                "instructions": current_layer.copy(),
                "layer_id": len(segments),
                "description": self._describe_layer(current_layer),
            }
            seg["complexity"] = self._compute_segment_complexity(seg)
            segments.append(seg)

        for i, instruction in enumerate(circuit.data):
            if instruction.operation.name in self.SKIP_OPS:
                continue
            current_layer.append(
                {
                    "instruction": instruction,
                    "index": i,
                    "operation": instruction.operation.name,
                    "qubits": [circuit.find_bit(q).index for q in instruction.qubits],
                    "params": getattr(instruction.operation, "params", []),
                }
            )
            if len(current_layer) >= self.MAX_LAYER_SIZE:
                flush()
                current_layer = []
        flush()
        return segments

    @staticmethod
    def _describe_layer(layer: List[Dict[str, Any]]) -> str:
        op_counts: Dict[str, int] = {}
        for inst in layer:
            op = inst["operation"]
            op_counts[op] = op_counts.get(op, 0) + 1
        return ", ".join([f"{cnt}×{op}" for op, cnt in op_counts.items()]) or "(empty)"

    @staticmethod
    def _compute_segment_complexity(segment: Dict[str, Any]) -> Dict[str, int]:
        complexity = {"total_gates": len(segment["instructions"]), "two_qubit_gates": 0, "single_qubit_gates": 0}
        for inst in segment["instructions"]:
            if len(inst["qubits"]) >= 2 or inst["operation"] in QuantumDeltaDebugger.TWO_QUBIT_GATES:
                complexity["two_qubit_gates"] += 1
            elif len(inst["qubits"]) == 1:
                complexity["single_qubit_gates"] += 1
        return complexity
    
    def _compute_total_complexity(self, segment_indices: Sequence[int]) -> Dict[str, int]:
        total = {"total_gates": 0, "two_qubit_gates": 0, "single_qubit_gates": 0}
        for idx in segment_indices:
            if 0 <= idx < len(self.segments):
                comp = self.segments[idx].get("complexity", {})
                for key in total:
                    total[key] += comp.get(key, 0)
        return total

    # ---------- circuit builders ----------
    def build_circuit_without_segments(
        self, original_circuit: QuantumCircuit, segments_to_exclude: Sequence[int]
    ) -> QuantumCircuit:
        try:
            new_circuit = original_circuit.copy_empty_like()
        except Exception:
            new_circuit = QuantumCircuit(original_circuit.num_qubits, original_circuit.num_clbits)

        excluded_indices = set()
        for seg_idx in segments_to_exclude:
            if 0 <= seg_idx < len(self.segments):
                for inst in self.segments[seg_idx]["instructions"]:
                    excluded_indices.add(inst["index"])

        for i, inst in enumerate(original_circuit.data):
            op = inst.operation
            qargs = inst.qubits
            if op.name in self.SKIP_OPS:
                continue
            if i in excluded_indices:
                continue
            q_indices = [original_circuit.find_bit(q).index for q in qargs]
            mapped_qargs = [new_circuit.qubits[idx] for idx in q_indices]
            new_circuit.append(op, mapped_qargs, [])

        for q_idx, c_idx in self.measure_map.items():
            new_circuit.measure(q_idx, c_idx)

        return new_circuit

    # ---------- evaluation ----------
    def evaluate_circuit(self, circuit: QuantumCircuit, shots: Optional[int] = None) -> Tuple[float, Dict[str, int], Dict[str, int]]:
        self.test_count += 1
        shots = shots or self.executor.config["execution"]["shots"]

        baseline = self.executor.run_circuit(circuit, execution_type=self.baseline_execution_type, shots=shots)
        if not baseline.get("success"):
            raise RuntimeError(f"Baseline execution failed: {baseline.get('error')}")
        baseline_counts = baseline.get("counts", {})

        test = self.executor.run_circuit(circuit, execution_type=self.test_execution_type, shots=shots)
        if not test.get("success"):
            test = self.executor.run_circuit(circuit, execution_type=self.test_execution_type, shots=shots)
        if not test.get("success"):
            raise RuntimeError(f"Test execution failed: {test.get('error')}")
        test_counts = test.get("counts", {})

        loss, _ = calculate_tvd(baseline_counts, test_counts)
        return loss, baseline_counts, test_counts

    # ---------- ddmin ----------
    def _test_exclusion(
        self, excluded: List[int], kept: List[int], action: str, prev_loss: float, shots: int,
    ) -> Dict[str, Any]:
        """Test excluding a set of segments. Returns log entry with loss and normalized_score."""
        circ = self.build_circuit_without_segments(self.original_circuit, excluded)
        loss, _, _ = self.evaluate_circuit(circ, shots=shots)

        complexity = self._compute_total_complexity(excluded)
        delta_loss = prev_loss - loss
        d2q = complexity["two_qubit_gates"]
        normalized_score = delta_loss / d2q if d2q > 0 else delta_loss / max(complexity["total_gates"], 1)

        entry = {
            "action": action,
            "excluded": excluded,
            "kept": kept,
            "shots": shots,
            "loss": loss,
            "prev_loss": prev_loss,
            "delta_loss": delta_loss,
            "complexity": complexity,
            "normalized_score": normalized_score,
            "progressed": False,
        }
        return entry

    def ddmin(self, baseline_loss: Optional[float] = None, initial_candidates: Optional[List[int]] = None) -> List[int]:
        if self.original_circuit is None:
            return []
        shots = self.executor.config["execution"]["shots"]
        if baseline_loss is None:
            full_circuit = self.build_circuit_without_segments(self.original_circuit, [])
            baseline_loss, _, _ = self.evaluate_circuit(full_circuit, shots=shots)
        loss = baseline_loss

        candidates = initial_candidates if initial_candidates is not None else list(range(len(self.segments)))
        self._current_candidates = candidates
        n = 2
        while len(candidates) >= 2:
            subsets = self._split(candidates, n)
            progressed = False

            for subset in subsets:
                complement = [i for i in candidates if i not in subset]
                if not complement:
                    continue

                for excluded, kept, action in [
                    (subset, complement, "test_subset"),
                    (complement, subset, "test_complement"),
                ]:
                    entry = self._test_exclusion(excluded, kept, action, loss, shots)
                    if entry["normalized_score"] > self.tolerance:
                        entry["progressed"] = True
                        self.ddmin_log.append(entry)
                        self._notify_step()
                        candidates = excluded
                        self._current_candidates = candidates
                        loss = entry["loss"]
                        progressed = True
                        break
                    self.ddmin_log.append(entry)
                    self._notify_step()

                if progressed:
                    break

            if not progressed:
                if n < min(self.max_granularity, len(candidates)):
                    n = min(n * 2, len(candidates))
                else:
                    break
            else:
                n = 2
        return candidates

    def _split(self, items: List[int], n: int) -> List[List[int]]:
        if n >= len(items):
            return [[x] for x in items]
        size = len(items) // n
        rem = len(items) % n
        out: List[List[int]] = []
        start = 0
        for i in range(n):
            end = start + size + (1 if i < rem else 0)
            if start < len(items):
                out.append(items[start:end])
            start = end
        return [s for s in out if s]

    def _notify_step(self):
        if self._on_step:
            self._on_step(self._build_result(self._current_candidates))

    def _build_result(self, candidates: List[int]) -> Dict[str, Any]:
        meta = {
            "timestamp": datetime.now().isoformat(),
            "circuit_info": {
                "total_qubits": self.original_circuit.num_qubits,
                "measured_qubits": self.measured_qubits_list,
            },
            "measurement_mapping": self.measure_map,
            "evaluation": {
                "shots": self.executor.config["execution"]["shots"],
                "tolerance": self.tolerance,
                "max_granularity": self.max_granularity,
                "test_mode": self.test_mode,
                "baseline_execution_type": self.baseline_execution_type,
                "test_execution_type": self.test_execution_type,
            },
        }
        return {
            "meta": meta,
            "problematic_segments": candidates,
            "segments_info": self.segments,
            "ddmin_log": self.ddmin_log,
        }

    # ---------- public API ----------
    def debug_circuit(
        self,
        circuit: QuantumCircuit,
        resume_candidates: Optional[List[int]] = None,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        self.original_circuit = circuit
        self._on_step = on_step

        # Extract measurement operations and mapping
        measured_qubits_set = set()
        measure_map_temp = {}
        for inst in circuit.data:
            if inst.operation.name == "measure":
                q_idx = circuit.find_bit(inst.qubits[0]).index
                c_idx = circuit.find_bit(inst.clbits[0]).index
                measured_qubits_set.add(q_idx)
                measure_map_temp[q_idx] = c_idx

        # Check that circuit has measurement operations
        if not measured_qubits_set:
            raise ValueError(
                "Circuit has no measurement operations. "
                "Please ensure the circuit includes measurements before running delta debug."
            )

        # Save sorted list of measured qubit indices and measurement mapping
        self.measured_qubits_list = sorted(list(measured_qubits_set))
        self.measure_map = measure_map_temp

        self.test_count = 0
        self.ddmin_log.clear()

        self.segments = self.extract_circuit_segments(circuit)

        # First get baseline loss for reporting
        full = self.build_circuit_without_segments(self.original_circuit, [])
        baseline_loss, _, _ = self.evaluate_circuit(full)
        # Record baseline evaluation
        self.ddmin_log.append({
            "action": "baseline",
            "excluded": [],
            "kept": list(range(len(self.segments))),
            "shots": self.executor.config["execution"]["shots"],
            "loss": baseline_loss,
        })

        self._current_candidates = list(range(len(self.segments)))
        minimal_set = self.ddmin(baseline_loss, initial_candidates=resume_candidates)
        self._current_candidates = minimal_set

        return self._build_result(minimal_set)

    # ---------- report ----------
    def save_debug_report(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        return filename


def run_delta_debug_on_isa(
    executor,
    isa_circuit: QuantumCircuit,
    tolerance: float = 0.01,
    test_mode: bool = False,
    max_granularity: int = 16,
    resume_candidates: Optional[List[int]] = None,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    dbg = QuantumDeltaDebugger(
        executor=executor,
        tolerance=tolerance,
        test_mode=test_mode,
        max_granularity=max_granularity
    )
    return dbg.debug_circuit(
        isa_circuit,
        resume_candidates=resume_candidates,
        on_step=on_step,
    )
