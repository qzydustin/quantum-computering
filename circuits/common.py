"""Shared utilities for circuit generation scripts."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quantum.runtime_ops import QuantumServiceManager
from quantum.artifacts import save_circuit_with_diagram


def transpile_and_save(
    source_circuit: QuantumCircuit,
    artifacts_dir: Path,
    name_prefix: str,
) -> tuple:
    """Connect to backend, transpile, and save QPY + PNG.

    Returns (isa_circuit, qpy_path, png_path).
    """
    svc = QuantumServiceManager(config_file="quantum_config.json")
    assert svc.connect(), "Failed to connect to IBM Quantum service"
    backend = svc.select_backend()
    assert backend is not None, "Failed to select backend"

    opt_level = svc.config["execution"]["optimization_level"]
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    isa = pm.run(source_circuit)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{name_prefix}_{backend.name}_{ts}"
    artifacts_dir.mkdir(exist_ok=True)
    qpy_path, png_path = save_circuit_with_diagram(isa, artifacts_dir, base_name)
    return isa, qpy_path, png_path
