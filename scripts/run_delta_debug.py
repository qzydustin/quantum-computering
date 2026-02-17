#!/usr/bin/env python3
"""
Delta Debugging Experiment Runner

This script only runs experiment steps:
1. Load circuit from artifacts directory
2. Optional initial execution test
3. Delta debugging and report generation

Post-analysis/verification/visualization are intentionally moved to notebooks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from qiskit import qpy

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum.quantum_executor import QuantumExecutor
from quantum.delta_debug import run_delta_debug_on_isa
from quantum.delta_debug_visualizer import generate_html_report


def load_circuit(artifacts_dir: str) -> Optional[object]:
    """Load the latest QPY circuit from an artifacts directory."""
    qpy_dir = Path(artifacts_dir)
    if not qpy_dir.is_absolute():
        qpy_dir = REPO_ROOT / qpy_dir
    if not qpy_dir.exists():
        print(f"❌ Directory not found: {artifacts_dir}")
        return None

    qpy_files = sorted(qpy_dir.glob('*.qpy'), reverse=True)
    if not qpy_files:
        print(f"❌ No .qpy files found in {artifacts_dir}")
        return None

    qpy_path = qpy_files[0]
    print(f"📁 Loading circuit: {qpy_path}")

    try:
        with open(qpy_path, 'rb') as f:
            isa = list(qpy.load(f))[0]
        print("✅ Circuit loaded")
        return isa
    except Exception as e:
        print(f"❌ Failed to load circuit: {e}")
        return None


def run_initial_test(isa_circuit, config_file: str = "quantum_config.json"):
    """Run optional initial execution test on the input ISA circuit."""
    print("\n" + "=" * 60)
    print("Step 1: Initial Execution Test")
    print("=" * 60)

    try:
        qe = QuantumExecutor(config_file=str((REPO_ROOT / config_file).resolve()))
        result = qe.run_circuit(isa_circuit=isa_circuit, execution_type="all")
        print("✅ Initial test completed")
        return result
    except Exception as e:
        print(f"⚠️  Initial test failed: {e}")
        return None


def run_delta_debugging(
    isa_circuit,
    output_dir: str,
    config_file: str = "quantum_config.json",
    tolerance: float = 0.01,
    max_granularity: int = 16,
    test_mode: bool = False,
) -> Optional[dict]:
    """Run delta debugging and generate timestamped JSON+HTML reports."""
    print("\n" + "=" * 60)
    print("Step 2: Delta Debugging")
    print("=" * 60)

    try:
        import os
        from datetime import datetime

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        original_cwd = os.getcwd()

        try:
            os.chdir(str(output_path))

            qe = QuantumExecutor(config_file=str((REPO_ROOT / config_file).resolve()))
            result = run_delta_debug_on_isa(
                executor=qe,
                isa_circuit=isa_circuit,
                tolerance=tolerance,
                max_granularity=max_granularity,
                test_mode=test_mode,
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = generate_html_report(result, output_path=f"delta_debug_report_{ts}.html")

            print("\n📊 Key Findings:")
            print(f"  • Total segments: {result['total_segments']}")
            print(f"  • Problematic segments: {len(result['problematic_segments'])}")
            print(f"  • Problematic segment IDs: {result['problematic_segments']}")
            print(f"  • Baseline loss: {result['baseline_loss']:.4f}")
            print(f"  • Test count: {result['test_count']}")
            print(f"\n✅ HTML report generated: {output_path / html_path}")
            return result
        finally:
            os.chdir(original_cwd)
    except Exception as e:
        print(f"❌ Delta debugging failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_delta_experiment(
    algorithm_dir: str,
    config_file: str = "quantum_config.json",
    skip_initial_test: bool = False,
    tolerance: float = 0.01,
    max_granularity: int = 16,
    test_mode: bool = False,
):
    """Run delta debug experiment only (no post-processing)."""
    print("=" * 60)
    print(f"🚀 Delta Debug Experiment: {algorithm_dir}")
    print("=" * 60)

    artifacts_path = Path(algorithm_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = REPO_ROOT / artifacts_path
    if not artifacts_path.exists():
        print(f"❌ Directory not found: {algorithm_dir}")
        return

    isa = load_circuit(str(artifacts_path))
    if isa is None:
        return

    if not skip_initial_test:
        run_initial_test(isa, config_file)

    debug_result = run_delta_debugging(
        isa,
        output_dir=str(artifacts_path),
        config_file=config_file,
        tolerance=tolerance,
        max_granularity=max_granularity,
        test_mode=test_mode,
    )

    if debug_result is None:
        print("❌ Delta debugging failed")
        return

    print("\n" + "=" * 60)
    print("✅ Experiment completed")
    print("Post-processing notebooks:")
    print("  - notebooks/sequence_loss_analysis.ipynb")
    print("  - notebooks/noisy_expected_vs_observed.ipynb")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run delta debugging experiment for quantum circuits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_delta_debug.py --algorithm artifacts/bv3-1
  python run_delta_debug.py --algorithm artifacts/qaoa3-1 --tolerance 0.02
  python run_delta_debug.py --algorithm artifacts/qft4-1 --skip-initial-test
        """,
    )

    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        required=True,
        help="Algorithm directory (e.g., artifacts/bv3-1 or bv3-1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="quantum_config.json",
        help="Path to quantum config file (default: quantum_config.json)",
    )
    parser.add_argument(
        "--skip-initial-test",
        action="store_true",
        help="Skip the initial execution test step",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance threshold for delta debugging (default: 0.01)",
    )
    parser.add_argument(
        "--max-granularity",
        type=int,
        default=16,
        help="Maximum splitting depth (default: 16)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (noisy vs ideal instead of real vs noisy)",
    )

    args = parser.parse_args()

    algorithm_dir = args.algorithm
    if not algorithm_dir.startswith("artifacts/"):
        algorithm_dir = f"artifacts/{algorithm_dir}"

    run_delta_experiment(
        algorithm_dir=algorithm_dir,
        config_file=args.config,
        skip_initial_test=args.skip_initial_test,
        tolerance=args.tolerance,
        max_granularity=args.max_granularity,
        test_mode=args.test_mode,
    )


if __name__ == "__main__":
    main()
