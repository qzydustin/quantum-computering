import json
from typing import Any, Dict, List, Optional
from qiskit_ibm_runtime import QiskitRuntimeService

class QuantumServiceManager:
    """IBM Quantum Service Manager"""
    
    def __init__(self, config_file="quantum_config.json"):
        """
        Initialize quantum service manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.service = None
        self.backend = None
        self.last_connect_error = None
        
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found")
            raise FileNotFoundError(f"Configuration file {config_file} is required")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {config_file}")
            raise ValueError(f"Invalid JSON format in {config_file}")
    
    def connect(self):
        """Connect to IBM Quantum service"""
        config = self.config["ibm_quantum"]
        token = config.get("token")
        channel = config.get("channel")
        instance = config.get("instance")

        # Try config as-is first.
        attempts: List[Dict[str, Any]] = [{
            "token": token,
            "channel": channel,
            "instance": instance,
        }]

        # For platform channel, instance may be optional depending on account setup.
        if channel == "ibm_quantum_platform":
            attempts.append({
                "token": token,
                "channel": channel,
            })

        # De-duplicate attempts.
        dedup = []
        seen = set()
        for a in attempts:
            key = (a.get("channel"), a.get("instance"))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(a)

        last_error = None
        for kwargs in dedup:
            try:
                self.service = QiskitRuntimeService(**kwargs)
                self.last_connect_error = None
                return True
            except Exception as e:
                last_error = e

        self.last_connect_error = last_error
        print(f"Failed to connect to IBM Quantum service: {last_error}")
        return False
    
    def select_backend(self):
        """
        Select backend device
        
        Returns:
            Selected backend device or None
        """
        if not self.service:
            return None
        
        backend_name = self.config["ibm_quantum"]["backend"]
        
        try:
            self.backend = self.service.backend(backend_name)
            return self.backend
        except Exception as e:
            print(f"Failed to select backend {backend_name}: {e}")
            return None

    def get_job(self, job_id: str):
        """Fetch one runtime job by ID."""
        if not self.service:
            return None
        try:
            return self.service.job(job_id)
        except Exception as e:
            print(f"Failed to get job {job_id}: {e}")
            return None

    def list_jobs(
        self,
        limit: int = 20,
        pending: Optional[bool] = None,
        backend_name: Optional[str] = None,
    ):
        """List runtime jobs with optional filters."""
        if not self.service:
            return []
        kwargs: Dict[str, Any] = {"limit": limit}
        if pending is not None:
            kwargs["pending"] = pending
        if backend_name:
            kwargs["backend_name"] = backend_name
        try:
            return self.service.jobs(**kwargs)
        except Exception as e:
            print(f"Failed to list jobs: {e}")
            return []

    def cancel_job(self, job_id: str) -> bool:
        """Cancel one runtime job by ID."""
        job = self.get_job(job_id)
        if not job:
            return False
        try:
            job.cancel()
            return True
        except Exception as e:
            print(f"Failed to cancel job {job_id}: {e}")
            return False
