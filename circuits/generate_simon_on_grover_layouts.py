#!/usr/bin/env python3
"""
Generate Simon circuits mapped onto the physical qubit layouts used by grover3-1 and grover3-2.

This keeps the Simon workload on the same local hardware neighborhoods so we can compare
whether the previously observed error patterns are layout-specific or algorithm-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simon_3_qubits import simon_3q_circuit
from remap_qpy_layouts import (
    load_first_qpy_circuit,
    get_active_qubits,
    build_compact_circuit,
    remap_compact_circuit,
    ensure_backend,
)
from quantum.artifacts import save_circuit_with_diagram


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


GROVER_LAYOUT_SPECS = {
    "grover3-1": {
        "source_qpy": REPO_ROOT / "artifacts/grover3-1/grover3_l3_ibm_fez_20251105_150131.qpy",
        # Keep the Simon target on the q143-centered neighborhood that repeatedly appeared in Grover3-1.
        "mapping": [136, 144, 143],  # logical input0, input1, output
    },
    "grover3-2": {
        "source_qpy": REPO_ROOT / "artifacts/grover3-2/grover3_l3_ibm_fez_20251203_175232.qpy",
        # Keep the Simon target on the q107-centered neighborhood highlighted in Grover3-2.
        "mapping": [106, 108, 107],  # logical input0, input1, output
    },
}


def generate_simon_for_layout(
    layout_name: str,
    secret: str,
    output_root: Path,
    config_path: Path,
    backend_name: str | None = None,
) -> None:
    spec = GROVER_LAYOUT_SPECS[layout_name]
    source_qpy = spec["source_qpy"]
    target_mapping = list(spec["mapping"])

    grover_isa = load_first_qpy_circuit(source_qpy)
    grover_active = get_active_qubits(grover_isa)
    config = load_config(config_path)
    opt_level = int(config["execution"]["optimization_level"])

    backend = ensure_backend(str(config_path), backend_name)
    actual_backend_name = getattr(backend, "name", None)
    actual_backend_name = actual_backend_name() if callable(actual_backend_name) else actual_backend_name

    simon_source = simon_3q_circuit(secret_string=secret)
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    simon_isa = pm.run(simon_source)
    simon_active = get_active_qubits(simon_isa)
    simon_compact = build_compact_circuit(simon_isa, simon_active)
    simon_mapped = remap_compact_circuit(simon_compact, target_mapping)

    out_dir = output_root / f"simon-on-{layout_name}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"simon3_{secret}_{layout_name}_{actual_backend_name}_{ts}"
    qpy_path, png_path = save_circuit_with_diagram(simon_mapped, out_dir, base_name)

    metadata = {
        "layout_name": layout_name,
        "backend": actual_backend_name,
        "secret": secret,
        "optimization_level": opt_level,
        "grover_source_qpy": str(source_qpy.relative_to(REPO_ROOT)),
        "grover_active_qubits": grover_active,
        "simon_transpiled_active_qubits": simon_active,
        "logical_to_physical": {
            "input0": target_mapping[0],
            "input1": target_mapping[1],
            "output": target_mapping[2],
        },
        "qpy_path": str(qpy_path.relative_to(REPO_ROOT)),
        "png_path": str(png_path.relative_to(REPO_ROOT)),
    }
    (out_dir / "mapping_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{layout_name}: {qpy_path}")
    print(f"  grover active qubits: {grover_active}")
    print(f"  simon logical->physical: {metadata['logical_to_physical']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Simon circuits on grover3-1 and grover3-2 qubit layouts")
    parser.add_argument("--secret", default="10", help="Simon secret string")
    parser.add_argument(
        "--layout",
        choices=["grover3-1", "grover3-2", "both"],
        default="both",
        help="Which target layout to generate",
    )
    parser.add_argument("--output", default="artifacts", help="Output root directory")
    parser.add_argument("--backend", default=None, help="Optional backend override")
    parser.add_argument("--config", default="quantum_config.json", help="Quantum config JSON path")
    args = parser.parse_args()

    layouts = ["grover3-1", "grover3-2"] if args.layout == "both" else [args.layout]
    output_root = Path(args.output)
    config_path = Path(args.config).expanduser().resolve()
    for layout_name in layouts:
        generate_simon_for_layout(
            layout_name=layout_name,
            secret=args.secret,
            output_root=output_root,
            config_path=config_path,
            backend_name=args.backend,
        )


if __name__ == "__main__":
    main()
