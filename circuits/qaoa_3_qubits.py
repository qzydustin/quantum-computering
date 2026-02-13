"""
QAOA 3-qubit Circuit Generator and Visualizer

This script generates and visualizes QAOA (Quantum Approximate Optimization Algorithm) 
circuits for 3 qubits based on QASMBench implementation.

Reference: QASMBench qaoa_n3
Cost function: C = -1 + z(0)z(2) - 2 z(0)z(1)z(2) - 3 z(1) with p = 1
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from ibm_quantum_connector import QuantumServiceManager
from quantum_utils import (
    load_existing_circuit,
    visualize_circuit,
    save_circuit_with_diagram
)


def qaoa_3q_circuit(gamma: Optional[float] = None, 
                    beta: Optional[float] = None) -> QuantumCircuit:
    """
    Create a QAOA circuit for 3 qubits following QASMBench qaoa_n3 implementation.
    
    The cost function is: C = -1 + z(0)z(2) - 2 z(0)z(1)z(2) - 3 z(1)
    With p = 1 (single layer)
    
    Args:
        gamma: Parameter for the cost Hamiltonian (default: 1.79986 * pi)
        beta: Parameter for the mixer Hamiltonian (default: 0.545344 * pi)
    
    Returns:
        QuantumCircuit: The QAOA circuit measuring all qubits.
    """
    n_qubits = 3
    
    # Default parameters from QASMBench qaoa_n3
    if gamma is None:
        gamma = 1.79986 * np.pi
    if beta is None:
        beta = 0.545344 * np.pi
    
    # Build circuit with named registers
    qreg = QuantumRegister(n_qubits, "q")
    creg = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qreg, creg)
    
    # 1) Initialization: create uniform superposition
    for q in range(n_qubits):
        qc.h(qreg[q])
    qc.barrier()
    
    # 2) Cost Hamiltonian (U_C)
    # Term: z(0)z(2) with coefficient 1
    qc.cx(qreg[0], qreg[2])
    qc.rz(gamma, qreg[2])
    qc.cx(qreg[0], qreg[2])
    
    # Term: z(0)z(1)z(2) with coefficient -2
    qc.cx(qreg[0], qreg[1])
    qc.cx(qreg[1], qreg[2])
    qc.rz(-2 * gamma, qreg[2])  # -3.59973 = -2 * 1.79986
    qc.cx(qreg[1], qreg[2])
    qc.cx(qreg[0], qreg[1])
    
    # Note: The z(1) term with coefficient -3 is absorbed into the mixer phase
    
    # 3) Mixer Hamiltonian (U_M) - X rotations
    # Following QASMBench exact order
    qc.rx(beta, qreg[2])
    qc.rz(-5.39959 * np.pi, qreg[1])  # Additional Z rotation
    qc.rx(beta, qreg[0])
    
    # 4) Measure qubits (following QASMBench order: q[2], q[0], q[1])
    qc.measure(qreg[2], creg[2])
    
    # Additional rx on q[1] before measurement (from QASMBench)
    qc.rx(beta, qreg[1])
    
    qc.measure(qreg[0], creg[0])
    qc.measure(qreg[1], creg[1])
    
    return qc


def generate_qaoa_circuit(artifacts_dir: Path = Path('artifacts'),
                          gamma: Optional[float] = None,
                          beta: Optional[float] = None) -> tuple:
    """
    Generate a new QAOA 3-qubit circuit and save it to artifacts.
    
    Args:
        artifacts_dir: Directory to save circuit files
        gamma: Parameter for the cost Hamiltonian
        beta: Parameter for the mixer Hamiltonian
        
    Returns:
        Tuple of (circuit, qpy_path, png_path)
    """
    artifacts_dir.mkdir(exist_ok=True)
    
    print('🔧 Generating new 3-qubit QAOA circuit...')
    
    # Connect and select backend
    svc = QuantumServiceManager(config_file='quantum_config.json')
    assert svc.connect(), 'Failed to connect to IBM Quantum service'
    backend = svc.select_backend()
    assert backend is not None, 'Failed to select backend'
    
    # Build source circuit and compile to L3
    src_qc = qaoa_3q_circuit(gamma=gamma, beta=beta)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(src_qc)
    
    # Generate timestamp for filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'qaoa3_l3_{backend.name}_{ts}'
    
    # Save circuit with diagram
    qpy_path, png_path = save_circuit_with_diagram(isa, artifacts_dir, base_name)
    
    return isa, qpy_path, png_path


def main(num_circuits: int = 1, 
         load_existing: bool = True,
         gamma: Optional[float] = None,
         beta: Optional[float] = None,
         artifacts_dir: str = 'artifacts',
         visualize: bool = True):
    """
    Main function to generate QAOA 3-qubit circuits.
    
    Args:
        num_circuits: Number of circuits to generate
        load_existing: Whether to try loading existing circuit first
        gamma: Parameter for the cost Hamiltonian (None = use default)
        beta: Parameter for the mixer Hamiltonian (None = use default)
        artifacts_dir: Directory to store artifacts
        visualize: Whether to generate visualization images
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    print(f'\n{"="*60}')
    print('🚀 QAOA 3-qubit Circuit Generator')
    print(f'{"="*60}')
    print(f'📊 Number of circuits to generate: {num_circuits}')
    print(f'📁 Artifacts directory: {artifacts_path.absolute()}')
    print(f'{"="*60}\n')
    
    circuits = []
    
    for i in range(num_circuits):
        print(f'\n--- Circuit {i+1}/{num_circuits} ---')
        
        if load_existing and i == 0:
            # Try to load existing circuit for first iteration
            result = load_existing_circuit('qaoa3_l3_*.qpy', artifacts_path)
            if result:
                circuit, qpy_path = result
                circuits.append((circuit, qpy_path, None))
                
                # Visualize if requested and no PNG exists
                if visualize:
                    png_path = qpy_path.with_suffix('.png')
                    if not png_path.exists():
                        visualize_circuit(circuit, png_path)
                    else:
                        print(f'🎨 Circuit diagram already exists: {png_path.name}')
                continue
        
        # Generate new circuit
        circuit, qpy_path, png_path = generate_qaoa_circuit(
            artifacts_dir=artifacts_path,
            gamma=gamma,
            beta=beta
        )
        circuits.append((circuit, qpy_path, png_path))
    
    print(f'\n{"="*60}')
    print(f'✨ Summary: Generated/Loaded {len(circuits)} circuit(s)')
    print(f'{"="*60}')
    
    for i, (circuit, qpy_path, png_path) in enumerate(circuits, 1):
        print(f'\nCircuit {i}:')
        print(f'  📄 QPY: {qpy_path.name}')
        if png_path:
            print(f'  🎨 PNG: {png_path.name}')
        print(f'  📏 Depth: {circuit.depth()}')
        print(f'  🔧 Gates: {circuit.size()}')
        print(f'  🎯 Qubits: {circuit.num_qubits}')
    
    print(f'\n{"="*60}')
    
    return circuits


if __name__ == '__main__':
    # Configuration parameters
    NUM_CIRCUITS = 1        # Number of circuits to generate
    LOAD_EXISTING = True     # Whether to load existing circuits
    GAMMA = None             # Cost parameter (None = use default from QASMBench)
    BETA = None              # Mixer parameter (None = use default from QASMBench)
    ARTIFACTS_DIR = 'artifacts'  # Output directory
    VISUALIZE = True         # Whether to generate visualization images

    # Run main function
    circuits = main(
        num_circuits=NUM_CIRCUITS,
        load_existing=LOAD_EXISTING,
        gamma=GAMMA,
        beta=BETA,
        artifacts_dir=ARTIFACTS_DIR,
        visualize=VISUALIZE
    )

