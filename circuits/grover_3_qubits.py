"""
Grover 3-qubit Circuit Generator and Visualizer

This script generates and visualizes Grover's algorithm circuits for 3 qubits.
It can generate multiple circuits and save them to the artifacts directory.

Reference: IBM Quantum Learning – Grover's algorithm
https://quantum.cloud.ibm.com/learning/en/courses/utility-scale-quantum-computing/grovers-algorithm
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


def grover_3q_circuit(marked_states: Optional[list] = None) -> QuantumCircuit:
    """
    Create a Grover search circuit for 3 qubits following IBM's reference implementation
    with an ancilla prepared in |->, an oracle via phase kickback, and the standard
    diffusion operator on the input register.

    Args:
        marked_states: List of integers representing the marked basis states.
            If None, defaults to marking the last basis state (7).

    Returns:
        QuantumCircuit: The Grover circuit measuring only the input register.
    """
    n_qubits = 3
    
    if marked_states is None:
        marked_states = [7]  # 2^3 - 1

    # Build circuit with named registers
    inp = QuantumRegister(n_qubits, "inp")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(n_qubits, "c")
    qc = QuantumCircuit(inp, anc, creg)

    # 1) Initialization: create uniform superposition on inputs; ancilla to |->
    for q in range(n_qubits):
        qc.h(inp[q])
    qc.x(anc[0])
    qc.h(anc[0])
    qc.barrier()

    # Helper: oracle via phase kickback on ancilla for a single marked bitstring
    def apply_oracle_for_bits(bitstring: str) -> None:
        # Align q0 with the rightmost (LSB) bit in Qiskit's little-endian ordering
        bits = bitstring[::-1]
        # Flip zeros so controls fire when state == target
        for idx, bit in enumerate(bits):
            if bit == '0':
                qc.x(inp[idx])
        # Multi-controlled X onto ancilla implements a phase flip via |->
        qc.mcx([inp[i] for i in range(n_qubits)], anc[0])
        # Uncompute the preparation
        for idx, bit in enumerate(bits):
            if bit == '0':
                qc.x(inp[idx])
        qc.barrier()

    # Helper: standard diffusion operator on input qubits
    def apply_diffusion() -> None:
        # H on inputs
        for q in range(n_qubits):
            qc.h(inp[q])
        # X on inputs
        for q in range(n_qubits):
            qc.x(inp[q])
        # Implement multi-controlled Z about |0...0>
        qc.h(inp[-1])
        qc.mcx([inp[i] for i in range(n_qubits - 1)], inp[-1])
        qc.h(inp[-1])
        # Uncompute X and finish H
        for q in range(n_qubits):
            qc.x(inp[q])
        for q in range(n_qubits):
            qc.h(inp[q])
        qc.barrier()

    # Optimal number of Grover iterations (fixed to 1 for current experiments)
    num_iterations = 1
    print(f"num_iterations: {num_iterations}")

    # 2) Grover iterations: Oracle then Diffusion
    for _ in range(num_iterations):
        for state in marked_states:
            bitstring = format(state, f'0{n_qubits}b')
            apply_oracle_for_bits(bitstring)
        apply_diffusion()

    # 3) Clear ancilla back to |0>
    qc.h(anc[0])
    qc.x(anc[0])
    qc.barrier()

    # 4) Measure only input qubits
    for i in range(n_qubits):
        qc.measure(inp[i], creg[i])

    return qc




def generate_grover_circuit(artifacts_dir: Path = Path('artifacts'),
                            marked_states: Optional[list] = None) -> tuple:
    """
    Generate a new Grover 3-qubit circuit and save it to artifacts.
    
    Args:
        artifacts_dir: Directory to save circuit files
        marked_states: States to mark in the oracle
        
    Returns:
        Tuple of (circuit, qpy_path, png_path)
    """
    artifacts_dir.mkdir(exist_ok=True)
    
    print('🔧 Generating new 3-qubit Grover circuit...')
    
    # Connect and select backend
    svc = QuantumServiceManager(config_file='quantum_config.json')
    assert svc.connect(), 'Failed to connect to IBM Quantum service'
    backend = svc.select_backend()
    assert backend is not None, 'Failed to select backend'
    
    # Build source circuit and compile to L3
    src_qc = grover_3q_circuit(marked_states=marked_states)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(src_qc)
    
    # Generate timestamp for filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'grover3_l3_{backend.name}_{ts}'
    
    # Save circuit with diagram
    qpy_path, png_path = save_circuit_with_diagram(isa, artifacts_dir, base_name)
    
    return isa, qpy_path, png_path




def main(num_circuits: int = 1, 
         load_existing: bool = True,
         marked_states: Optional[list] = None,
         artifacts_dir: str = 'artifacts',
         visualize: bool = True):
    """
    Main function to generate Grover 3-qubit circuits.
    
    Args:
        num_circuits: Number of circuits to generate
        load_existing: Whether to try loading existing circuit first
        marked_states: States to mark in the oracle (None = mark last state)
        artifacts_dir: Directory to store artifacts
        visualize: Whether to generate visualization images
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    print(f'\n{"="*60}')
    print('🚀 Grover 3-qubit Circuit Generator')
    print(f'{"="*60}')
    print(f'📊 Number of circuits to generate: {num_circuits}')
    print(f'📁 Artifacts directory: {artifacts_path.absolute()}')
    print(f'{"="*60}\n')
    
    circuits = []
    
    for i in range(num_circuits):
        print(f'\n--- Circuit {i+1}/{num_circuits} ---')
        
        if load_existing and i == 0:
            # Try to load existing circuit for first iteration
            result = load_existing_circuit('grover3_l3_*.qpy', artifacts_path)
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
        circuit, qpy_path, png_path = generate_grover_circuit(
            artifacts_dir=artifacts_path,
            marked_states=marked_states
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
    MARKED_STATES = None     # Marked states (None marks the last state: 7)
    ARTIFACTS_DIR = 'artifacts'  # Output directory
    VISUALIZE = True         # Whether to generate visualization images

    # Run main function
    circuits = main(
        num_circuits=NUM_CIRCUITS,
        load_existing=LOAD_EXISTING,
        marked_states=MARKED_STATES,
        artifacts_dir=ARTIFACTS_DIR,
        visualize=VISUALIZE
    )
