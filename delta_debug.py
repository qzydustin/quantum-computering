from __future__ import annotations

from typing import List, Dict, Any, Optional, Sequence, Tuple
from datetime import datetime
import json

from qiskit import QuantumCircuit


class QuantumDeltaDebugger:
    """
    Quantum Circuit Delta Debugger (based on DDMin)

    - Targets `QuantumCircuit` compiled to backend ISA
    - Measures "loss" using Total Variation Distance between baseline and test distributions
    - Uses DDMin to find the minimal set of circuit segments causing the maximum loss
    """

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
        
        # Set execution types based on mode
        # Normal mode: test real_device against noisy_simulator baseline
        # Test mode: test noisy_simulator against ideal_simulator baseline
        if self.test_mode:
            self.test_execution_type = "noisy_simulator"
            self.baseline_execution_type = "ideal_simulator"
        else:
            self.test_execution_type = "real_device"
            self.baseline_execution_type = "noisy_simulator"

        self.original_circuit: Optional[QuantumCircuit] = None
        self.segments: List[Dict[str, Any]] = []
        self.measured_qubits_list: List[int] = []  # List of measured qubit indices
        self.measure_map: Dict[int, int] = {}  # Mapping: qubit_index -> clbit_index
        self.test_count: int = 0
        self.ddmin_log: List[Dict[str, Any]] = []
        self.evaluations: List[Dict[str, Any]] = []

    # ---------- segmentation ----------
    def extract_circuit_segments(self, circuit: QuantumCircuit) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        current_layer: List[Dict[str, Any]] = []
        for i, instruction in enumerate(circuit.data):
            if instruction.operation.name in ["measure", "barrier", "delay", "reset"]:
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
            if self._should_end_layer(current_layer, instruction):
                if current_layer:
                    seg = {
                        "instructions": current_layer.copy(),
                        "layer_id": len(segments),
                        "description": self._describe_layer(current_layer),
                    }
                    seg["complexity"] = self._compute_segment_complexity(seg)
                    segments.append(seg)
                    current_layer = []
        if current_layer:
            seg = {
                "instructions": current_layer.copy(),
                "layer_id": len(segments),
                "description": self._describe_layer(current_layer),
            }
            seg["complexity"] = self._compute_segment_complexity(seg)
            segments.append(seg)
        return segments

    def _should_end_layer(self, current_layer: List[Dict[str, Any]], instruction) -> bool:
        if len(current_layer) >= 5:
            return True
        if instruction.operation.name in ["cx", "cz", "mcx", "ccx"]:
            return True
        # If a conditional gate is encountered, cut the layer to avoid crossing conditional boundaries
        if getattr(instruction.operation, "condition", None) is not None:
            return True
        return False

    def _describe_layer(self, layer: List[Dict[str, Any]]) -> str:
        op_counts: Dict[str, int] = {}
        for inst in layer:
            op = inst["operation"]
            op_counts[op] = op_counts.get(op, 0) + 1
        return ", ".join([f"{cnt}×{op}" for op, cnt in op_counts.items()]) or "(empty)"
    
    def _compute_segment_complexity(self, segment: Dict[str, Any]) -> Dict[str, int]:
        """
        Compute complexity metrics for a segment
        
        Returns:
            Dictionary containing various complexity metrics
        """
        # List of two-qubit gates (common two-qubit and multi-controlled gates)
        two_qubit_gates = [
            # Basic two-qubit gates
            "cx", "cy", "cz", "ch", "swap", "iswap", "dcx", "ecr",
            # Parameterized two-qubit gates
            "rzz", "rxx", "ryy", "rzx",
            # Controlled rotation gates
            "crx", "cry", "crz", "cp", "cu", "cu1", "cu3",
            # Multi-controlled gates
            "mcx", "mcy", "mcz", "ccx", "ccy", "ccz",
        ]
        
        complexity = {
            "total_gates": len(segment["instructions"]),
            "two_qubit_gates": 0,
            "single_qubit_gates": 0,
        }
        
        for inst in segment["instructions"]:
            op_name = inst["operation"]
            n_qubits = len(inst["qubits"])
            
            if n_qubits >= 2 or op_name in two_qubit_gates:
                complexity["two_qubit_gates"] += 1
            elif n_qubits == 1:
                complexity["single_qubit_gates"] += 1
        
        return complexity
    
    def _compute_total_complexity(self, segment_indices: Sequence[int]) -> Dict[str, int]:
        """
        Compute total complexity for multiple segments
        
        Args:
            segment_indices: List of segment indices
            
        Returns:
            Total complexity metrics
        """
        total = {
            "total_gates": 0,
            "two_qubit_gates": 0,
            "single_qubit_gates": 0,
        }
        
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

        # Extract original measurement mappings
        original_measurements = []
        for i, inst in enumerate(original_circuit.data):
            if inst.operation.name == "measure":
                q_idx = original_circuit.find_bit(inst.qubits[0]).index
                c_idx = original_circuit.find_bit(inst.clbits[0]).index
                original_measurements.append((q_idx, c_idx))

        # Require original circuit to have measurements
        if not original_measurements:
            raise ValueError(
                "Original circuit has no measurement operations. "
                "Please ensure the circuit includes measurements before running delta debug."
            )

        for i, inst in enumerate(original_circuit.data):
            op = inst.operation
            qargs = inst.qubits
            # Skip measurement and non-execution instructions
            if getattr(op, "name", None) in ["measure", "barrier", "delay", "reset"]:
                continue
            if i in excluded_indices:
                continue
            try:
                q_indices = [original_circuit.find_bit(q).index for q in qargs]
                mapped_qargs = [new_circuit.qubits[idx] for idx in q_indices]
            except Exception:
                mapped_qargs = [new_circuit.qubits[j] for j in range(len(qargs))]
            new_circuit.append(op, mapped_qargs, [])

        # Apply measurements: preserve original mappings only
        for q_idx, c_idx in original_measurements:
            new_circuit.measure(q_idx, c_idx)
        
        # Record measurement mapping (qubit -> clbit)
        self.measure_map = {q_idx: c_idx for q_idx, c_idx in original_measurements}
        
        return new_circuit

    # ---------- evaluation ----------
    def evaluate_circuit(self, circuit: QuantumCircuit, shots: int = 2048) -> Tuple[float, Dict[str, int], Dict[str, int]]:
        self.test_count += 1

        def _project_counts(counts: Dict[str, int], measured_qubits: List[int]) -> Dict[str, int]:
            """
            Project measurement results to specified qubits
            
            Args:
                counts: Full measurement counts dict, key is bitstring (in clbit order)
                measured_qubits: List of measured qubit indices (sorted)
                
            Returns:
                Counts dict containing only measured qubits
                
            Note:
                Qiskit counts bitstring is ordered by clbits, typically little-endian (right side is LSB)
                Need to map qubit indices to clbit indices via measure_map
            """
            from collections import defaultdict
            acc = defaultdict(int)
            
            # Compute clbit index sequence (in qubit order)
            clbits_in_order = [self.measure_map[q] for q in measured_qubits]
            
            for k, v in (counts or {}).items():
                bits = k.replace(" ", "")
                # Qiskit uses little-endian: bits[::-1][c] gets the value of clbit c
                try:
                    projected_bits = ''.join(bits[::-1][c] for c in clbits_in_order)
                    acc[projected_bits] += v
                except IndexError:
                    # Skip counts with mismatched length
                    continue
            return dict(acc)

        # Use measured qubits from the circuit
        measured_qubits = sorted(self.measured_qubits_list)

        # Run baseline execution
        baseline = self.executor.run_circuit(circuit, execution_type=self.baseline_execution_type, shots=shots)
        if not baseline.get("success"):
            raise RuntimeError(f"Baseline execution failed: {baseline.get('error')}")
        baseline_counts = _project_counts(baseline.get("counts", {}), measured_qubits)
        total_baseline = sum(baseline_counts.values())
        
        # Check that counts after projection are not empty
        if total_baseline == 0:
            raise RuntimeError(
                f"Baseline execution produced empty counts after projection. "
                f"Raw counts: {baseline.get('counts', {})}, measured_qubits: {measured_qubits}"
            )

        # Run test execution
        test = self.executor.run_circuit(circuit, execution_type=self.test_execution_type, shots=shots)
        if not test.get("success"):
            # Retry once; if still fails, throw error to avoid treating failure as 0 probability
            test = self.executor.run_circuit(circuit, execution_type=self.test_execution_type, shots=shots)
        if not test.get("success"):
            raise RuntimeError(f"Test execution failed: {test.get('error')}")
        test_counts = _project_counts(test.get("counts", {}), measured_qubits)
        total_test = sum(test_counts.values())
        
        # Check that counts after projection are not empty
        if total_test == 0:
            raise RuntimeError(
                f"Test execution produced empty counts after projection. "
                f"Raw counts: {test.get('counts', {})}, measured_qubits: {measured_qubits}"
            )

        # Calculate loss as sum of absolute differences across all possible states
        # This measures the total variation distance between distributions
        all_states = set(baseline_counts.keys()) | set(test_counts.keys())
        loss = 0.0
        for state in all_states:
            baseline_prob = baseline_counts.get(state, 0) / total_baseline
            test_prob = test_counts.get(state, 0) / total_test
            loss += abs(baseline_prob - test_prob)

        return loss, baseline_counts, test_counts

    # ---------- ddmin ----------
    def ddmin(self, baseline_loss: Optional[float] = None) -> List[int]:
        if self.original_circuit is None:
            return []
        shots = 2048
        full_circuit = self.build_circuit_without_segments(self.original_circuit, [])
        full_loss, _, _ = self.evaluate_circuit(full_circuit, shots=shots)
        loss = full_loss if baseline_loss is None else baseline_loss
        # Clear ddmin log
        self.ddmin_log.clear()

        candidates = list(range(len(self.segments)))
        n = 2
        while len(candidates) >= 2:
            subsets = self._split(candidates, n)
            progressed = False
            
            for subset in subsets:
                complement = [i for i in candidates if i not in subset]
                if not complement:
                    continue
                
                # Compute complexity change after excluding subset
                subset_complexity = self._compute_total_complexity(subset)
                delta_2q = subset_complexity["two_qubit_gates"]
                
                test_circ = self.build_circuit_without_segments(self.original_circuit, subset)
                t_loss, _, _ = self.evaluate_circuit(test_circ, shots=shots)
                
                # Compute loss change and normalized score
                delta_loss = loss - t_loss
                # Avoid division by zero: if no two-qubit gates, use total gates
                if delta_2q > 0:
                    normalized_score = delta_loss / delta_2q
                else:
                    delta_gates = subset_complexity["total_gates"]
                    normalized_score = delta_loss / max(delta_gates, 1)
                
                # Record ddmin evaluation
                self.evaluations.append({
                    "mode": "ddmin",
                    "excluded": subset,
                    "shots": shots,
                    "loss": t_loss,
                    "delta_loss": delta_loss,
                    "complexity": subset_complexity,
                    "normalized_score": normalized_score,
                })
                log_entry = {
                    "action": "test_subset",
                    "excluded": subset,
                    "kept": complement,
                    "loss": t_loss,
                    "prev_loss": loss,
                    "delta_loss": delta_loss,
                    "complexity": subset_complexity,
                    "normalized_score": normalized_score,
                    "progressed": False,
                }
                # Use normalized score for judgment: if loss reduction per unit complexity exceeds threshold, consider significant
                if normalized_score > self.tolerance:
                    candidates = subset
                    loss = t_loss
                    log_entry["progressed"] = True
                    self.ddmin_log.append(log_entry)
                    progressed = True
                    break
                # Try testing complement
                comp_circ = self.build_circuit_without_segments(self.original_circuit, complement)
                c_loss, _, _ = self.evaluate_circuit(comp_circ, shots=shots)
                
                # Compute complement's complexity change and normalized score
                complement_complexity = self._compute_total_complexity(complement)
                delta_2q_comp = complement_complexity["two_qubit_gates"]
                delta_loss_comp = loss - c_loss
                
                if delta_2q_comp > 0:
                    normalized_score_comp = delta_loss_comp / delta_2q_comp
                else:
                    delta_gates_comp = complement_complexity["total_gates"]
                    normalized_score_comp = delta_loss_comp / max(delta_gates_comp, 1)
                
                # Record ddmin evaluation (complement)
                self.evaluations.append({
                    "mode": "ddmin",
                    "excluded": complement,
                    "shots": shots,
                    "loss": c_loss,
                    "delta_loss": delta_loss_comp,
                    "complexity": complement_complexity,
                    "normalized_score": normalized_score_comp,
                })
                log_entry_comp = {
                    "action": "test_complement",
                    "excluded": complement,
                    "kept": subset,
                    "loss": c_loss,
                    "prev_loss": loss,
                    "delta_loss": delta_loss_comp,
                    "complexity": complement_complexity,
                    "normalized_score": normalized_score_comp,
                    "progressed": False,
                }
                # Use normalized score for judgment
                if normalized_score_comp > self.tolerance:
                    candidates = complement
                    loss = c_loss
                    log_entry_comp["progressed"] = True
                    self.ddmin_log.append(log_entry_comp)
                    progressed = True
                    break
                # Both did not progress, record both logs
                self.ddmin_log.append(log_entry)
                self.ddmin_log.append(log_entry_comp)
            if not progressed:
                # Limit maximum granularity to avoid excessive splitting
                if n < min(self.max_granularity, len(candidates)):
                    n = min(n * 2, len(candidates))
                else:
                    break
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

    # ---------- public API ----------
    def debug_circuit(
        self,
        circuit: QuantumCircuit,
    ) -> Dict[str, Any]:
        self.original_circuit = circuit
        
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
        self.evaluations.clear()

        self.segments = self.extract_circuit_segments(circuit)
        
        # Identify ancilla qubits (unmeasured qubits)
        ancilla_qubits = set(range(circuit.num_qubits)) - measured_qubits_set

        # First get baseline loss for reporting
        full = self.build_circuit_without_segments(self.original_circuit, [])
        baseline_loss, _, _ = self.evaluate_circuit(full)
        # Record baseline evaluation
        self.evaluations.append({
            "mode": "baseline",
            "excluded": [],
            "included": list(range(len(self.segments))),
            "shots": 2048,
            "loss": baseline_loss,
        })

        minimal_set = self.ddmin(baseline_loss)

        analysis = self.analyze_problematic_segments(minimal_set)
        meta = {
            "timestamp": datetime.now().isoformat(),
            "circuit_info": {
                "total_qubits": circuit.num_qubits,
                "measured_qubits": self.measured_qubits_list,
                "measured_qubits_count": len(self.measured_qubits_list),
                "ancilla_qubits": sorted(list(ancilla_qubits)),
                "ancilla_count": len(ancilla_qubits),
            },
            "bit_endianness": "Qiskit counts bitstring indexed by clbits (little-endian typical); we index with bits[::-1][clbit]",
            "measurement_mapping": {f"qubit_{q}": f"clbit_{c}" for q, c in self.measure_map.items()},
            "loss_calculation": f"sum of |p_baseline(state) - p_test(state)| over all states of measured qubits {self.measured_qubits_list} (Total Variation Distance)",
            "evaluation": {
                "shots_ddmin": 2048,
                "tolerance": self.tolerance,
                "tolerance_description": "Normalized threshold: (Δloss / Δtwo_qubit_gates) > tolerance → significant contribution",
                "normalization_metric": "two_qubit_gates (fallback to total_gates if no two-qubit gates)",
                "max_granularity": self.max_granularity,
                "max_granularity_description": "Maximum number of subsets to split candidates into, limits DDMin splitting depth",
                "test_retry": 1,
                "test_mode": self.test_mode,
                "baseline_execution_type": self.baseline_execution_type,
                "test_execution_type": self.test_execution_type,
            },
        }

        return {
            "meta": meta,
            "total_segments": len(self.segments),
            "problematic_segments": minimal_set,
            "analysis": analysis,
            "baseline_loss": baseline_loss,
            "test_count": self.test_count,
            "segments_info": self.segments,
            "ddmin_log": self.ddmin_log,
            "evaluations": self.evaluations,
        }

    def analyze_problematic_segments(self, problematic_segments: Sequence[int]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "segment_count": len(problematic_segments),
            "segments": [],
            "operation_analysis": {},
            "qubit_analysis": {},
            "complexity_analysis": {},
        }
        ops: List[str] = []
        qubits: set[int] = set()
        total_complexity = {
            "total_gates": 0,
            "two_qubit_gates": 0,
            "single_qubit_gates": 0,
        }
        
        for idx in problematic_segments:
            if 0 <= idx < len(self.segments):
                seg = self.segments[idx]
                complexity = seg.get("complexity", {})
                
                # Accumulate total complexity
                for key in total_complexity:
                    total_complexity[key] += complexity.get(key, 0)
                
                info = {
                    "layer_id": seg["layer_id"],
                    "description": seg["description"],
                    "instructions": len(seg["instructions"]),
                    "complexity": complexity,
                    "operations": [],
                }
                for inst in seg["instructions"]:
                    info["operations"].append(
                        {
                            "operation": inst["operation"],
                            "qubits": inst["qubits"],
                            "params": inst["params"],
                        }
                    )
                    ops.append(inst["operation"])
                    qubits.update(inst["qubits"])
                out["segments"].append(info)
        
        from collections import Counter
        out["operation_analysis"] = dict(Counter(ops))
        out["qubit_analysis"] = {
            "affected_qubits": sorted(qubits),
            "qubit_count": len(qubits),
        }
        out["complexity_analysis"] = total_complexity
        return out

    # ---------- report ----------
    def save_debug_report(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"delta_debug_report_{ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        return filename


def run_delta_debug_on_isa(
    executor,
    isa_circuit: QuantumCircuit,
    tolerance: float = 0.01,
    test_mode: bool = False,
    max_granularity: int = 16,
) -> Dict[str, Any]:
    """
    Run Delta Debug analysis
    
    Args:
        executor: Quantum circuit executor
        isa_circuit: ISA-compiled quantum circuit
        tolerance: Tolerance threshold for DDMin algorithm
        test_mode: If False (default), test real_device against noisy_simulator baseline.
                   If True, test noisy_simulator against ideal_simulator baseline.
        max_granularity: Maximum number of subsets to split candidates into (default: 16)
    
    Returns:
        Debug result dictionary
    """
    dbg = QuantumDeltaDebugger(
        executor=executor,
        tolerance=tolerance,
        test_mode=test_mode,
        max_granularity=max_granularity
    )
    result = dbg.debug_circuit(isa_circuit)
    # Automatically save report
    dbg.save_debug_report(result)
    return result
