from __future__ import annotations
from typing import Dict, Optional, Sequence, List
import json
import time

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler

from .runtime_ops import QuantumServiceManager
from .metrics import calculate_tvd


class QuantumExecutor:
    """Quantum Circuit Executor with sync and async real-device support."""

    def __init__(self, config_file="quantum_config.json"):
        with open(config_file, "r") as f:
            self.config = json.load(f)
        self.service_manager = QuantumServiceManager(config_file=config_file)
        assert self.service_manager.connect(), "Failed to connect to IBM Quantum service"
        self.backend = self.service_manager.select_backend()
        assert self.backend is not None, "Failed to select backend"

        self.ideal_sim = AerSimulator()
        self.noisy_sim = AerSimulator.from_backend(self.backend)
        self._job_service_by_id: Dict[str, object] = {}

    # ---- helpers ----

    @property
    def _shots(self) -> int:
        return self.config["execution"]["shots"]

    @property
    def _optimization_level(self) -> int:
        return self.config["execution"]["optimization_level"]

    @property
    def _backend_name(self) -> Optional[str]:
        return getattr(self.backend, "name", None) if self.backend else None

    @staticmethod
    def _extract_counts(result) -> Dict[str, int]:
        """Extract counts from PrimitiveResult via DataBin -> BitArray.get_counts()."""
        data = result[0].data
        for bit_array in data.values():
            if hasattr(bit_array, "get_counts"):
                return bit_array.get_counts()
        return {}

    def _make_pub(self, isa_circuit: QuantumCircuit, shots: int,
                  param_vals: Optional[Sequence[float]] = None):
        params = [] if not isa_circuit.parameters else (param_vals or [0.0] * len(isa_circuit.parameters))
        return (isa_circuit, params, shots)

    def _refresh_runtime_account(self) -> bool:
        """Re-select account/backend before each real-device submission."""
        if not self.service_manager.connect():
            return False
        backend = self.service_manager.select_backend()
        if backend is None:
            return False

        self.backend = backend
        self.noisy_sim = AerSimulator.from_backend(self.backend)
        return True

    def _get_job(self, job_id: str):
        """Fetch a runtime job by ID from tracked service or active service."""
        service = self._job_service_by_id.get(job_id) or self.service_manager.service
        if service is None:
            print(f"Failed to get job {job_id}: runtime service unavailable")
            return None
        try:
            return service.job(job_id)
        except Exception as e:
            print(f"Failed to get job {job_id}: {e}")
            return None

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile a circuit to ISA using config optimization_level."""
        pm = generate_preset_pass_manager(optimization_level=self._optimization_level, backend=self.backend)
        return pm.run(circuit)

    # ---- sync execution ----

    def run_circuit(self, isa_circuit: Optional[QuantumCircuit] = None,
                    execution_type: Optional[str] = None,
                    shots: Optional[int] = None,
                    param_vals: Optional[Sequence[float]] = None) -> dict:
        if execution_type is None:
            raise RuntimeError("execution_type is required")
        if isa_circuit is None:
            raise RuntimeError("ISA circuit is required")
        shots = shots or self._shots
        pub = self._make_pub(isa_circuit, shots, param_vals)

        if execution_type in ("ideal_simulator", "noisy_simulator"):
            sim = self.ideal_sim if execution_type == "ideal_simulator" else self.noisy_sim
            sampler = AerSampler.from_backend(sim)
            counts = self._extract_counts(sampler.run([pub]).result())
            return {"success": True, "execution_type": execution_type, "backend": self._backend_name,
                    "job_id": None, "counts": counts, "shots": shots}

        elif execution_type == "real_device":
            if not self._refresh_runtime_account():
                return {
                    "success": False,
                    "execution_type": "real_device",
                    "backend": self._backend_name,
                    "job_id": None,
                    "error": "Failed to refresh runtime account/backend",
                    "shots": shots,
                }
            sampler = RuntimeSampler(mode=self.backend)
            job = sampler.run([pub])
            job_id = job.job_id() if hasattr(job, "job_id") else None
            if job_id:
                self._job_service_by_id[job_id] = self.service_manager.service
            try:
                counts = self._extract_counts(job.result())
            except Exception as e:
                return {"success": False, "execution_type": "real_device", "backend": self._backend_name,
                        "job_id": job_id, "error": str(e), "shots": shots}
            return {"success": True, "execution_type": "real_device", "backend": self._backend_name,
                    "job_id": job_id, "counts": counts, "shots": shots}

        elif execution_type == "all":
            results = {}
            for et in ("ideal_simulator", "noisy_simulator", "real_device"):
                results[et] = self.run_circuit(isa_circuit, et, shots, param_vals)
            tvd_loss, state_details = calculate_tvd(
                results["noisy_simulator"]["counts"], results["real_device"]["counts"])
            return {
                "ideal": results["ideal_simulator"],
                "noisy": results["noisy_simulator"],
                "real": results["real_device"],
                "distribution_difference": {
                    "tvd_loss": tvd_loss, "state_details": state_details,
                    "description": "Total Variation Distance between noisy simulator and real device",
                },
            }
        else:
            return {"success": False, "error": f"Unknown execution_type: {execution_type}"}

    # ---- async real device: submit / poll / collect ----

    def submit_real_job(self, isa_circuit: QuantumCircuit, shots: Optional[int] = None,
                        param_vals: Optional[Sequence[float]] = None) -> dict:
        """Submit a real-device job without blocking. Returns job metadata."""
        shots = shots or self._shots
        if not self._refresh_runtime_account():
            return {
                "success": False,
                "job_id": None,
                "status": "ERROR",
                "shots": shots,
                "backend": self._backend_name,
                "error": "Failed to refresh runtime account/backend",
            }
        sampler = RuntimeSampler(mode=self.backend)
        job = sampler.run([self._make_pub(isa_circuit, shots, param_vals)])
        job_id = job.job_id()
        self._job_service_by_id[job_id] = self.service_manager.service
        return {"success": True, "job_id": job_id, "status": str(job.status()),
                "shots": shots, "backend": self._backend_name}

    def submit_real_jobs(self, circuits: Sequence[QuantumCircuit], shots: Optional[int] = None,
                         param_vals_list: Optional[Sequence[Optional[Sequence[float]]]] = None) -> List[dict]:
        """Submit multiple real-device jobs. Returns list of job metadata."""
        pv_list = param_vals_list or [None] * len(circuits)
        return [self.submit_real_job(c, shots, pv) for c, pv in zip(circuits, pv_list)]

    def poll_job(self, job_id: str) -> dict:
        """Check status of a submitted job."""
        job = self._get_job(job_id)
        if job is None:
            return {"job_id": job_id, "status": "NOT_FOUND", "done": False}
        return {"job_id": job_id, "status": str(job.status()), "done": job.done()}

    def wait_for_jobs(self, job_ids: Sequence[str], poll_interval: float = 10.0,
                      timeout: Optional[float] = None) -> List[dict]:
        """Block until all jobs reach a final state."""
        pending = set(job_ids)
        results: Dict[str, dict] = {}
        elapsed = 0.0
        while pending:
            for jid in list(pending):
                info = self.poll_job(jid)
                results[jid] = info
                if info["done"] or info["status"] in ("DONE", "ERROR", "CANCELLED", "NOT_FOUND"):
                    pending.discard(jid)
            if not pending:
                break
            if timeout is not None and elapsed >= timeout:
                break
            time.sleep(poll_interval)
            elapsed += poll_interval
        return [results[jid] for jid in job_ids]

    def collect_result(self, job_id: str, shots: Optional[int] = None) -> dict:
        """Retrieve result of a completed job by ID."""
        job = self._get_job(job_id)
        if job is None:
            return {"success": False, "error": f"Job not found: {job_id}"}
        try:
            counts = self._extract_counts(job.result())
            shots = shots or self._shots
            return {"success": True, "execution_type": "real_device", "backend": self._backend_name,
                    "job_id": job_id, "counts": counts, "shots": shots}
        except Exception as e:
            return {"success": False, "job_id": job_id, "error": str(e)}

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a runtime job by ID."""
        job = self._get_job(job_id)
        if job is None:
            return {"success": False, "job_id": job_id}
        try:
            job.cancel()
            return {"success": True, "job_id": job_id}
        except Exception as e:
            print(f"Failed to cancel job {job_id}: {e}")
            return {"success": False, "job_id": job_id}
