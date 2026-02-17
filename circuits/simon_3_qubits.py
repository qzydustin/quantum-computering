"""
Simon 3-qubit Circuit Generator and Visualizer

This script generates and visualizes Simon's algorithm circuits for 3 qubits
(2 input qubits + 1 output qubit) based on QASMBench implementation.

Reference: QASMBench simon_n6 (simplified to 3 qubits)
Secret string: s = 10 (2-bit case)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quantum.ibm_quantum_connector import QuantumServiceManager
from quantum.circuit_artifacts import (
    load_existing_circuit,
    visualize_circuit,
    save_circuit_with_diagram
)


def simon_3q_circuit(secret_string: Optional[str] = None) -> QuantumCircuit:
    """
    Create a Simon's algorithm circuit for 3 qubits (2 input + 1 output).
    
    Args:
        secret_string: 2-bit secret string (e.g., "10", "11", "01"). 
                      If None, defaults to "10".
    
    Returns:
        QuantumCircuit: The Simon circuit measuring input qubits.
    """
    n_input = 2
    n_output = 1
    
    if secret_string is None:
        secret_string = "10"
    
    if len(secret_string) != n_input:
        raise ValueError(f"Secret string must be {n_input} bits long")
    
    # Build circuit with named registers
    input_reg = QuantumRegister(n_input, "input")
    output_reg = QuantumRegister(n_output, "output")
    creg = ClassicalRegister(n_input, "c")
    qc = QuantumCircuit(input_reg, output_reg, creg)
    
    # 1) Initialization: create uniform superposition on input qubits
    for q in range(n_input):
        qc.h(input_reg[q])
    qc.barrier()
    
    # 2) Oracle: Apply the secret structure
    # For secret string s = "10" (binary), we need f(x) = f(x ⊕ s)
    # This is implemented using Toffoli gates
    # Based on QASMBench simon_n6 structure, adapted for 3 qubits
    
    # Prepare output qubit
    qc.x(output_reg[0])
    
    # Apply oracle based on secret string
    # For s = "10": f(00) = f(10), f(01) = f(11)
    # We use Toffoli gates to implement the function
    if secret_string == "10":
        # f(x) = x[0] XOR (x[0] AND x[1]) for s=10
        qc.cx(input_reg[1], output_reg[0])  # x[1] affects output
        qc.ccx(input_reg[0], input_reg[1], output_reg[0])  # Toffoli
    elif secret_string == "11":
        # f(x) = x[0] XOR x[1] for s=11
        qc.cx(input_reg[0], output_reg[0])
        qc.cx(input_reg[1], output_reg[0])
    elif secret_string == "01":
        # f(x) = x[0] for s=01
        qc.cx(input_reg[0], output_reg[0])
    else:
        # Default: same as "10"
        qc.cx(input_reg[1], output_reg[0])
        qc.ccx(input_reg[0], input_reg[1], output_reg[0])
    
    qc.barrier()
    
    # 3) Apply Hadamard to input qubits (before measurement)
    for q in range(n_input):
        qc.h(input_reg[q])
    
    qc.barrier()
    
    # 4) Measure only input qubits
    for i in range(n_input):
        qc.measure(input_reg[i], creg[i])
    
    return qc


def generate_simon_circuit(artifacts_dir: Path = Path('artifacts'),
                          secret_string: Optional[str] = None) -> tuple:
    """
    Generate a new Simon 3-qubit circuit and save it to artifacts.
    
    Args:
        artifacts_dir: Directory to save circuit files
        secret_string: 2-bit secret string
        
    Returns:
        Tuple of (circuit, qpy_path, png_path)
    """
    artifacts_dir.mkdir(exist_ok=True)
    
    print('🔧 Generating new 3-qubit Simon circuit...')
    
    # Connect and select backend
    svc = QuantumServiceManager(config_file='quantum_config.json')
    assert svc.connect(), 'Failed to connect to IBM Quantum service'
    backend = svc.select_backend()
    assert backend is not None, 'Failed to select backend'
    
    # Build source circuit and compile to L3
    src_qc = simon_3q_circuit(secret_string=secret_string)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(src_qc)
    
    # Generate timestamp for filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'simon3_l3_{backend.name}_{ts}'
    
    # Save circuit with diagram
    qpy_path, png_path = save_circuit_with_diagram(isa, artifacts_dir, base_name)
    
    return isa, qpy_path, png_path


def main(num_circuits: int = 1, 
         load_existing: bool = True,
         secret_string: Optional[str] = None,
         artifacts_dir: str = 'artifacts',
         visualize: bool = True):
    """
    Main function to generate Simon 3-qubit circuits.
    
    Args:
        num_circuits: Number of circuits to generate
        load_existing: Whether to try loading existing circuit first
        secret_string: 2-bit secret string (None = "10")
        artifacts_dir: Directory to store artifacts
        visualize: Whether to generate visualization images
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    print(f'\n{"="*60}')
    print('🚀 Simon 3-qubit Circuit Generator')
    print(f'{"="*60}')
    print(f'📊 Number of circuits to generate: {num_circuits}')
    print(f'📁 Artifacts directory: {artifacts_path.absolute()}')
    print(f'{"="*60}\n')
    
    circuits = []
    
    for i in range(num_circuits):
        print(f'\n--- Circuit {i+1}/{num_circuits} ---')
        
        if load_existing and i == 0:
            # Try to load existing circuit for first iteration
            result = load_existing_circuit('simon3_l3_*.qpy', artifacts_path)
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
        circuit, qpy_path, png_path = generate_simon_circuit(
            artifacts_dir=artifacts_path,
            secret_string=secret_string
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
    SECRET_STRING = "10"     # 2-bit secret string (e.g., "10", "11", "01")
    ARTIFACTS_DIR = 'artifacts'  # Output directory
    VISUALIZE = True         # Whether to generate visualization images

    # Run main function
    circuits = main(
        num_circuits=NUM_CIRCUITS,
        load_existing=LOAD_EXISTING,
        secret_string=SECRET_STRING,
        artifacts_dir=ARTIFACTS_DIR,
        visualize=VISUALIZE
    )
