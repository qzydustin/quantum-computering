"""
Grover 3-qubit Circuit Generator and Visualizer

This script generates and visualizes Grover's algorithm circuits for 3 qubits.
It can generate multiple circuits and save them to the artifacts directory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

from ibm_quantum_connector import QuantumServiceManager
from grover_algorithm import grover_algorithm


def load_existing_circuit(n_qubits: int = 3, artifacts_dir: Path = Path('artifacts')) -> Optional[tuple]:
    """
    Load the most recent existing circuit from artifacts directory.
    
    Args:
        n_qubits: Number of qubits
        artifacts_dir: Directory containing circuit files
        
    Returns:
        Tuple of (circuit, qpy_path) if found, None otherwise
    """
    # Look for existing QPY files matching the pattern
    qpy_files = sorted(artifacts_dir.glob(f'grover{n_qubits}_l3_*.qpy'), reverse=True)
    
    if qpy_files:
        qpy_path = qpy_files[0]
        print(f'📁 Found existing circuit: {qpy_path.name}')
        with open(qpy_path, 'rb') as f:
            isa = list(qpy.load(f))[0]
        print('✅ Loaded existing ISA circuit')
        return isa, qpy_path
    
    return None


def generate_grover_circuit(n_qubits: int = 3, 
                            artifacts_dir: Path = Path('artifacts'),
                            marked_states: Optional[list] = None) -> tuple:
    """
    Generate a new Grover circuit and save it to artifacts.
    
    Args:
        n_qubits: Number of qubits
        artifacts_dir: Directory to save circuit files
        marked_states: States to mark in the oracle
        
    Returns:
        Tuple of (circuit, qpy_path, png_path)
    """
    artifacts_dir.mkdir(exist_ok=True)
    
    print(f'🔧 Generating new {n_qubits}-qubit Grover circuit...')
    
    # Connect and select backend
    svc = QuantumServiceManager(config_file='quantum_config.json')
    assert svc.connect(), 'Failed to connect to IBM Quantum service'
    backend = svc.select_backend()
    assert backend is not None, 'Failed to select backend'
    
    # Build source circuit and compile to L3
    src_qc = grover_algorithm(n_qubits, marked_states=marked_states)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa = pm.run(src_qc)
    
    # Generate timestamp for filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'grover{n_qubits}_l3_{backend.name}_{ts}'
    
    # Save ISA QPY
    qpy_path = artifacts_dir / f'{base_name}.qpy'
    with open(qpy_path, 'wb') as f:
        qpy.dump([isa], f)
    print(f'✅ Saved ISA circuit to: {qpy_path}')
    
    # Generate and save circuit diagram
    png_path = artifacts_dir / f'{base_name}.png'
    fig = circuit_drawer(isa, output='mpl', style='iqp', fold=-1)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'🎨 Saved circuit diagram to: {png_path}')
    
    return isa, qpy_path, png_path


def visualize_circuit(circuit, output_path: Optional[Path] = None, show: bool = False):
    """
    Visualize a quantum circuit.
    
    Args:
        circuit: Qiskit QuantumCircuit to visualize
        output_path: Path to save the image (optional)
        show: Whether to display the plot interactively
    """
    fig = circuit_drawer(circuit, output='mpl', style='iqp', fold=-1)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'🎨 Saved circuit diagram to: {output_path}')
    
    if show:
        plt.show()
    else:
        plt.close()


def main(num_circuits: int = 1, 
         n_qubits: int = 3, 
         load_existing: bool = True,
         marked_states: Optional[list] = None,
         artifacts_dir: str = 'artifacts',
         visualize: bool = True):
    """
    Main function to generate Grover circuits.
    
    Args:
        num_circuits: Number of circuits to generate
        n_qubits: Number of qubits for Grover algorithm
        load_existing: Whether to try loading existing circuit first
        marked_states: States to mark in the oracle (None = mark last state)
        artifacts_dir: Directory to store artifacts
        visualize: Whether to generate visualization images
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    print(f'\n{"="*60}')
    print(f'🚀 Grover {n_qubits}-qubit Circuit Generator')
    print(f'{"="*60}')
    print(f'📊 Number of circuits to generate: {num_circuits}')
    print(f'📁 Artifacts directory: {artifacts_path.absolute()}')
    print(f'{"="*60}\n')
    
    circuits = []
    
    for i in range(num_circuits):
        print(f'\n--- Circuit {i+1}/{num_circuits} ---')
        
        if load_existing and i == 0:
            # Try to load existing circuit for first iteration
            result = load_existing_circuit(n_qubits, artifacts_path)
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
            n_qubits=n_qubits,
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
    NUM_CIRCUITS = 10        # Number of circuits to generate
    N_QUBITS = 3            # Number of qubits
    LOAD_EXISTING = True    # Whether to load existing circuits
    MARKED_STATES = None    # Marked states (None marks the last state)
    ARTIFACTS_DIR = 'artifacts'  # Output directory
    VISUALIZE = True        # Whether to generate visualization images

    # Run main function
    circuits = main(
        num_circuits=NUM_CIRCUITS,
        n_qubits=N_QUBITS,
        load_existing=LOAD_EXISTING,
        marked_states=MARKED_STATES,
        artifacts_dir=ARTIFACTS_DIR,
        visualize=VISUALIZE
    )
