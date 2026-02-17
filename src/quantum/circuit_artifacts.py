from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from qiskit import qpy
from qiskit.visualization import circuit_drawer


def load_circuit_from_qpy(qpy_path: Path) -> Optional[object]:
    """Load the first quantum circuit from a QPY file."""
    try:
        with open(qpy_path, 'rb') as f:
            for circuit in qpy.load(f):
                return circuit
        return None
    except Exception as e:
        print(f"Error loading circuit from {qpy_path}: {e}")
        return None


def save_circuit_to_qpy(circuit, qpy_path: Path) -> bool:
    """Save a single circuit to a QPY file."""
    try:
        qpy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qpy_path, 'wb') as f:
            qpy.dump([circuit], f)
        return True
    except Exception as e:
        print(f"Error saving circuit to {qpy_path}: {e}")
        return False


def load_existing_circuit(pattern: str, artifacts_dir: Path = Path('artifacts')) -> Optional[tuple]:
    """Load the most recent QPY file matching a pattern under artifacts_dir."""
    qpy_files = sorted(artifacts_dir.glob(pattern), reverse=True)
    if not qpy_files:
        return None

    qpy_path = qpy_files[0]
    print(f'📁 Found existing circuit: {qpy_path.name}')
    circuit = load_circuit_from_qpy(qpy_path)
    if circuit is None:
        return None

    print('✅ Loaded existing ISA circuit')
    return circuit, qpy_path


def visualize_circuit(
    circuit,
    output_path: Optional[Path] = None,
    show: bool = False,
    style: str = 'iqp',
    fold: int = -1,
    dpi: int = 300,
):
    """Render and optionally save a circuit diagram."""
    fig = circuit_drawer(circuit, output='mpl', style=style, fold=fold)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f'🎨 Saved circuit diagram to: {output_path}')

    if show:
        plt.show()
    plt.close(fig)


def save_circuit_with_diagram(circuit, base_path: Path, base_name: str, dpi: int = 300) -> tuple:
    """Save a circuit as both QPY and PNG diagram."""
    base_path.mkdir(parents=True, exist_ok=True)

    qpy_path = base_path / f'{base_name}.qpy'
    save_circuit_to_qpy(circuit, qpy_path)
    print(f'✅ Saved ISA circuit to: {qpy_path}')

    png_path = base_path / f'{base_name}.png'
    visualize_circuit(circuit, png_path, dpi=dpi)

    return qpy_path, png_path
