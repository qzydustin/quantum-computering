"""
QFT 4-qubit Circuit Generator and Visualizer

This script generates and visualizes Quantum Fourier Transform (QFT) circuits 
for 4 qubits based on QASMBench implementation.

Reference: QASMBench qft_n4
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


def qft_4q_circuit(input_state: Optional[list] = None) -> QuantumCircuit:
    """
    Create a Quantum Fourier Transform circuit for 4 qubits following QASMBench qft_n4.
    
    Args:
        input_state: List of qubit indices to apply X gates (to prepare input state).
                   If None, defaults to [0, 2] as in QASMBench.
    
    Returns:
        QuantumCircuit: The QFT circuit measuring all qubits.
    """
    n_qubits = 4
    
    if input_state is None:
        input_state = [0, 2]  # Default from QASMBench qft_n4
    
    # Build circuit with named registers
    qreg = QuantumRegister(n_qubits, "q")
    creg = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(qreg, creg)
    
    # 1) Prepare input state (apply X gates to specified qubits)
    for idx in input_state:
        if 0 <= idx < n_qubits:
            qc.x(qreg[idx])
    qc.barrier()
    
    # 2) Quantum Fourier Transform
    # Apply QFT: H gates and controlled phase rotations
    for i in range(n_qubits):
        # Apply Hadamard to qubit i
        qc.h(qreg[i])
        
        # Apply controlled phase rotations
        for j in range(i + 1, n_qubits):
            # Controlled phase rotation: R_k where k = j - i + 1
            k = j - i + 1
            phase = np.pi / (2 ** k)
            qc.cp(phase, qreg[j], qreg[i])  # cp is controlled-phase (cu1)
    
    qc.barrier()
    
    # 3) Measure all qubits
    qc.measure_all()
    
    return qc


def generate_qft_circuit(artifacts_dir: Path = Path('artifacts'),
                        input_state: Optional[list] = None) -> tuple:
    """
    Generate a new QFT 4-qubit circuit and save it to artifacts.
    
    Args:
        artifacts_dir: Directory to save circuit files
        input_state: List of qubit indices to apply X gates
        
    Returns:
        Tuple of (circuit, qpy_path, png_path)
    """
    artifacts_dir.mkdir(exist_ok=True)
    
    print('🔧 Generating new 4-qubit QFT circuit...')
    
    # Connect and select backend
    svc = QuantumServiceManager(config_file='quantum_config.json')
    assert svc.connect(), 'Failed to connect to IBM Quantum service'
    backend = svc.select_backend()
    assert backend is not None, 'Failed to select backend'
    
    # Build source circuit and compile to L3
    src_qc = qft_4q_circuit(input_state=input_state)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(src_qc)
    
    # Generate timestamp for filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'qft4_l3_{backend.name}_{ts}'
    
    # Save circuit with diagram
    qpy_path, png_path = save_circuit_with_diagram(isa, artifacts_dir, base_name)
    
    return isa, qpy_path, png_path


def main(num_circuits: int = 1, 
         load_existing: bool = True,
         input_state: Optional[list] = None,
         artifacts_dir: str = 'artifacts',
         visualize: bool = True):
    """
    Main function to generate QFT 4-qubit circuits.
    
    Args:
        num_circuits: Number of circuits to generate
        load_existing: Whether to try loading existing circuit first
        input_state: List of qubit indices to apply X gates (None = [0, 2])
        artifacts_dir: Directory to store artifacts
        visualize: Whether to generate visualization images
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    print(f'\n{"="*60}')
    print('🚀 QFT 4-qubit Circuit Generator')
    print(f'{"="*60}')
    print(f'📊 Number of circuits to generate: {num_circuits}')
    print(f'📁 Artifacts directory: {artifacts_path.absolute()}')
    print(f'{"="*60}\n')
    
    circuits = []
    
    for i in range(num_circuits):
        print(f'\n--- Circuit {i+1}/{num_circuits} ---')
        
        if load_existing and i == 0:
            # Try to load existing circuit for first iteration
            result = load_existing_circuit('qft4_l3_*.qpy', artifacts_path)
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
        circuit, qpy_path, png_path = generate_qft_circuit(
            artifacts_dir=artifacts_path,
            input_state=input_state
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
    INPUT_STATE = [0, 2]     # Qubit indices to apply X gates (None = [0, 2])
    ARTIFACTS_DIR = 'artifacts'  # Output directory
    VISUALIZE = True         # Whether to generate visualization images

    # Run main function
    circuits = main(
        num_circuits=NUM_CIRCUITS,
        load_existing=LOAD_EXISTING,
        input_state=INPUT_STATE,
        artifacts_dir=ARTIFACTS_DIR,
        visualize=VISUALIZE
    )

