"""
Bernstein-Vazirani 19-qubit Circuit Generator and Visualizer

This script generates and visualizes Bernstein-Vazirani algorithm circuits for 19 qubits
(18 input qubits + 1 ancilla qubit) based on QASMBench bv_n19 implementation.

Reference: QASMBench bv_n19
Hidden string: s (18-bit string, default: "111111111111111111")
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def bv_19q_circuit(hidden_string: Optional[str] = None) -> QuantumCircuit:
    """
    Create a Bernstein-Vazirani algorithm circuit for 19 qubits (18 input + 1 ancilla).
    
    The algorithm finds a hidden string s by querying an oracle function f(x) = s·x mod 2.
    Based on QASMBench bv_n19 implementation.
    
    Args:
        hidden_string: 18-bit hidden string (e.g., "111111111111111111", "101010101010101010"). 
                      If None, defaults to "111111111111111111" (all ones).
    
    Returns:
        QuantumCircuit: The BV circuit measuring input qubits.
    """
    n_input = 18
    n_ancilla = 1
    
    if hidden_string is None:
        hidden_string = "111111111111111111"  # Default: all ones
    
    if len(hidden_string) != n_input:
        raise ValueError(f"Hidden string must be {n_input} bits long")
    
    # Build circuit with named registers
    input_reg = QuantumRegister(n_input, "input")
    ancilla_reg = QuantumRegister(n_ancilla, "ancilla")
    creg = ClassicalRegister(n_input, "c")
    qc = QuantumCircuit(input_reg, ancilla_reg, creg)
    
    # 1) Initialization: create uniform superposition on input qubits
    for q in range(n_input):
        qc.h(input_reg[q])
    
    # Prepare ancilla in |-> state
    qc.x(ancilla_reg[0])
    qc.h(ancilla_reg[0])
    qc.barrier()
    
    # 2) Oracle: Apply CNOT gates based on hidden string
    # For each bit '1' in the hidden string, apply CNOT from corresponding input to ancilla
    for i, bit in enumerate(hidden_string):
        if bit == '1':
            qc.cx(input_reg[i], ancilla_reg[0])
    
    qc.barrier()
    
    # 3) Apply Hadamard to input qubits (before measurement)
    for q in range(n_input):
        qc.h(input_reg[q])
    
    # Note: In QASMBench bv_n19, ancilla is not uncomputed before measurement
    # We measure only input qubits
    
    # 4) Measure only input qubits
    for i in range(n_input):
        qc.measure(input_reg[i], creg[i])
    
    return qc


def generate_bv_circuit(artifacts_dir: Path = Path('artifacts'),
                       hidden_string: Optional[str] = None) -> tuple:
    """
    Generate a new Bernstein-Vazirani 19-qubit circuit and save it to artifacts.
    
    Args:
        artifacts_dir: Directory to save circuit files
        hidden_string: 18-bit hidden string
        
    Returns:
        Tuple of (circuit, qpy_path, png_path)
    """
    artifacts_dir.mkdir(exist_ok=True)
    
    print('🔧 Generating new 19-qubit Bernstein-Vazirani circuit...')
    
    # Connect and select backend
    svc = QuantumServiceManager(config_file='quantum_config.json')
    assert svc.connect(), 'Failed to connect to IBM Quantum service'
    backend = svc.select_backend()
    assert backend is not None, 'Failed to select backend'
    
    # Build source circuit and compile to L3
    src_qc = bv_19q_circuit(hidden_string=hidden_string)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(src_qc)
    
    # Generate timestamp for filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'bv19_l3_{backend.name}_{ts}'
    
    # Save circuit with diagram
    qpy_path, png_path = save_circuit_with_diagram(isa, artifacts_dir, base_name)
    
    return isa, qpy_path, png_path


def main(num_circuits: int = 1, 
         load_existing: bool = True,
         hidden_string: Optional[str] = None,
         artifacts_dir: str = 'artifacts',
         visualize: bool = True):
    """
    Main function to generate Bernstein-Vazirani 19-qubit circuits.
    
    Args:
        num_circuits: Number of circuits to generate
        load_existing: Whether to try loading existing circuit first
        hidden_string: 18-bit hidden string (None = "111111111111111111")
        artifacts_dir: Directory to store artifacts
        visualize: Whether to generate visualization images
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    print(f'\n{"="*60}')
    print('🚀 Bernstein-Vazirani 19-qubit Circuit Generator')
    print(f'{"="*60}')
    print(f'📊 Number of circuits to generate: {num_circuits}')
    print(f'📁 Artifacts directory: {artifacts_path.absolute()}')
    print(f'{"="*60}\n')
    
    circuits = []
    
    for i in range(num_circuits):
        print(f'\n--- Circuit {i+1}/{num_circuits} ---')
        
        if load_existing and i == 0:
            # Try to load existing circuit for first iteration
            result = load_existing_circuit('bv19_l3_*.qpy', artifacts_path)
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
        circuit, qpy_path, png_path = generate_bv_circuit(
            artifacts_dir=artifacts_path,
            hidden_string=hidden_string
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
    HIDDEN_STRING = "111111111111111111"  # 18-bit hidden string (default: all ones)
    ARTIFACTS_DIR = 'artifacts'  # Output directory
    VISUALIZE = True         # Whether to generate visualization images

    # Run main function
    circuits = main(
        num_circuits=NUM_CIRCUITS,
        load_existing=LOAD_EXISTING,
        hidden_string=HIDDEN_STRING,
        artifacts_dir=ARTIFACTS_DIR,
        visualize=VISUALIZE
    )
