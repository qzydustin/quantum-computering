#!/usr/bin/env python3
"""
Delta Debugging Pipeline for Quantum Circuits

This script encapsulates the complete delta debugging workflow:
1. Load circuit from artifacts directory
2. Run initial execution test (optional)
3. Run delta debugging analysis
4. Analyze problematic segments and generate circuits
5. Verify circuits by comparing noisy simulator vs real device
6. Visualize results

Usage:
    python run_delta_debug.py --algorithm bv3-1
    python run_delta_debug.py --algorithm qaoa3-1 --skip-initial-test
    python run_delta_debug.py --algorithm grover3-3 --tolerance 0.02
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
from quantum.analyze_problematic_segments import analyze_and_generate_circuits
from quantum import verify_circuits


def load_circuit(artifacts_dir: str) -> Optional[object]:
    """
    Load the latest QPY circuit from artifacts directory.
    
    Args:
        artifacts_dir: Directory containing QPY files (e.g., 'artifacts/bv3-1')
        
    Returns:
        QuantumCircuit or None if not found
    """
    qpy_dir = Path(artifacts_dir)
    if not qpy_dir.is_absolute():
        qpy_dir = REPO_ROOT / qpy_dir
    if not qpy_dir.exists():
        print(f"❌ Directory not found: {artifacts_dir}")
        return None
    
    import glob
    qpy_files = sorted(glob.glob(str(qpy_dir / '*.qpy')), reverse=True)
    if not qpy_files:
        print(f"❌ No .qpy files found in {artifacts_dir}")
        return None
    
    qpy_path = qpy_files[0]
    print(f'📁 Loading circuit: {qpy_path}')
    
    try:
        with open(qpy_path, 'rb') as f:
            isa = list(qpy.load(f))[0]
        print('✅ Circuit loaded')
        return isa
    except Exception as e:
        print(f"❌ Failed to load circuit: {e}")
        return None


def run_initial_test(isa_circuit, config_file: str = 'quantum_config.json'):
    """
    Run initial execution test on the circuit (optional step).
    
    Args:
        isa_circuit: The ISA circuit to test
        config_file: Path to quantum config file
    """
    print("\n" + "="*60)
    print("Step 1: Initial Execution Test")
    print("="*60)
    
    try:
        qe = QuantumExecutor(config_file=str((REPO_ROOT / config_file).resolve()))
        result = qe.run_circuit(isa_circuit=isa_circuit, execution_type='all')
        print("✅ Initial test completed")
        return result
    except Exception as e:
        print(f"⚠️  Initial test failed: {e}")
        return None


def run_delta_debugging(
    isa_circuit,
    output_dir: str,
    config_file: str = 'quantum_config.json',
    tolerance: float = 0.01,
    max_granularity: int = 16,
    test_mode: bool = False
) -> Optional[dict]:
    """
    Run delta debugging analysis on the circuit.
    
    Args:
        isa_circuit: The ISA circuit to analyze
        output_dir: Directory to save reports (e.g., 'artifacts/bv3-1')
        config_file: Path to quantum config file
        tolerance: Tolerance threshold for delta debugging
        max_granularity: Maximum splitting depth
        test_mode: Whether to run in test mode
        
    Returns:
        Delta debugging result dictionary or None if failed
    """
    print("\n" + "="*60)
    print("Step 2: Delta Debugging")
    print("="*60)
    
    try:
        import os
        from datetime import datetime
        
        # Save reports to output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        original_cwd = os.getcwd()
        
        try:
            # Change to output directory so reports are saved there
            os.chdir(str(output_path))
            
            # Note: Reports are saved with timestamps (delta_debug_report_YYYYMMDD_HHMMSS.json)
            # This means old reports are never deleted or overwritten, allowing historical analysis.
            # All reports in this directory will be analyzed together in the next step.
            
            qe = QuantumExecutor(config_file=str((REPO_ROOT / config_file).resolve()))
            result = run_delta_debug_on_isa(
                executor=qe,
                isa_circuit=isa_circuit,
                tolerance=tolerance,
                max_granularity=max_granularity,
                test_mode=test_mode
            )
            
            # Generate HTML report in output directory
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"delta_debug_report_{ts}.html"
            html_path = generate_html_report(result, output_path=html_path)
            
            # Display key findings
            print("\n📊 Key Findings:")
            print(f"  • Total segments: {result['total_segments']}")
            print(f"  • Problematic segments: {len(result['problematic_segments'])}")
            print(f"  • Problematic segment IDs: {result['problematic_segments']}")
            print(f"  • Baseline loss: {result['baseline_loss']:.4f}")
            print(f"  • Test count: {result['test_count']}")
            print(f"\n✅ HTML report generated: {output_path / html_path}")
            
            return result
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    except Exception as e:
        print(f"❌ Delta debugging failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_segments(algorithm_dir: str, output_dir: str = "tmp") -> Optional[dict]:
    """
    Analyze problematic segments and generate circuits.
    
    This function analyzes ALL delta debug reports in the algorithm directory,
    allowing you to aggregate findings across multiple runs.
    
    Args:
        algorithm_dir: Algorithm-specific directory containing delta debug reports (e.g., 'artifacts/bv3-1')
        output_dir: Directory to save analysis results (default: 'tmp')
        
    Returns:
        Analysis results dictionary or None if failed
        
    Note:
        - Circuits are named seq_001, seq_002, etc. based on Top 10 ranking
        - These files will overwrite existing seq_XXX files (expected behavior)
        - sequence_analysis.json is always overwritten with latest analysis
    """
    print("\n" + "="*60)
    print("Step 4: Analyze Problematic Segments")
    print("="*60)
    print("📊 Analyzing ALL delta debug reports in the algorithm directory")
    print(f"   Directory: {algorithm_dir}")
    
    try:
        # analyze_and_generate_circuits searches for reports in the specified directory
        # It will find ALL delta_debug_report_*.json files in the algorithm directory
        # This allows analyzing all historical reports for this specific algorithm
        analysis = analyze_and_generate_circuits(algorithm_dir, str(REPO_ROOT / output_dir))
        
        # Copy circuits directory to algorithm directory
        import shutil
        tmp_circuits = REPO_ROOT / output_dir / 'circuits'
        if tmp_circuits.exists() and any(tmp_circuits.iterdir()):
            algorithm_path = Path(algorithm_dir)
            algorithm_path.mkdir(parents=True, exist_ok=True)
            dest_circuits = algorithm_path / 'circuits'
            dest_circuits.mkdir(parents=True, exist_ok=True)
            
            # Copy all circuit files (seq_XXX files will be overwritten)
            copied_count = 0
            for src_file in tmp_circuits.iterdir():
                if src_file.is_file():
                    dest_file = dest_circuits / src_file.name
                    shutil.copy2(src_file, dest_file)
                    copied_count += 1
            
            print(f"✅ Copied {copied_count} circuit file(s) to: {dest_circuits}")
            print("   Note: seq_XXX files overwritten with current Top 10 analysis results")
        
        # Copy sequence_analysis.json to algorithm directory (overwrites existing)
        tmp_analysis = REPO_ROOT / output_dir / 'sequence_analysis.json'
        if tmp_analysis.exists():
            algorithm_path = Path(algorithm_dir)
            algorithm_path.mkdir(parents=True, exist_ok=True)
            dest_analysis = algorithm_path / 'sequence_analysis.json'
            shutil.copy2(tmp_analysis, dest_analysis)
            print(f"✅ Copied sequence_analysis.json to: {dest_analysis}")
        
        return analysis
    except Exception as e:
        print(f"❌ Segment analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_circuits_workflow(algorithm_dir: str, config_file: str = 'quantum_config.json'):
    """
    Verify circuits by comparing noisy simulator vs real device.
    
    Args:
        algorithm_dir: Directory to save results to (e.g., 'artifacts/bv3-1')
        
    Note:
        Overwrites circuit_comparison_results.json with latest verification results
    """
    print("\n" + "="*60)
    print("Step 5: Verify Circuits")
    print("="*60)
    print("Compare all circuit segments using noisy simulator vs real device and calculate TVD loss")
    
    try:
        verify_circuits.verify_circuits(
            config_file=str((REPO_ROOT / config_file).resolve()),
            circuits_dir=str(REPO_ROOT / 'tmp' / 'circuits'),
            output_file=str(REPO_ROOT / 'tmp' / 'circuit_comparison_results.json'),
        )
        print("✅ Circuit verification completed")
        
        # Copy circuit_comparison_results.json to algorithm directory (overwrites existing)
        import shutil
        tmp_results = REPO_ROOT / 'tmp' / 'circuit_comparison_results.json'
        if tmp_results.exists():
            algorithm_path = Path(algorithm_dir)
            algorithm_path.mkdir(parents=True, exist_ok=True)
            dest_results = algorithm_path / 'circuit_comparison_results.json'
            shutil.copy2(tmp_results, dest_results)
            print(f"✅ Copied circuit_comparison_results.json to: {dest_results}")
    except Exception as e:
        print(f"❌ Circuit verification failed: {e}")
        import traceback
        traceback.print_exc()


def visualize_results_workflow(algorithm_dir: str):
    """
    Visualize verification results and save to algorithm directory.
    
    Args:
        algorithm_dir: Directory to save visualization files (e.g., 'artifacts/bv3-1')
        
    Note:
        Overwrites tvd_analysis.png and tvd_summary.csv with latest visualization
    """
    print("\n" + "="*60)
    print("Step 6: Visualize Results")
    print("="*60)
    
    try:
        import matplotlib
        # Set non-interactive backend to prevent opening windows
        matplotlib.use('Agg')
        
        from quantum.visualize_results import (
            load_results,
            visualize_tvd_with_sequences,
            print_detailed_summary
        )
        import pandas as pd
        
        print("🔍 Loading results...")
        comparison_results, sequence_analysis = load_results(str(REPO_ROOT / 'tmp'))
        
        if not comparison_results or not sequence_analysis:
            print("❌ Failed to load results. Please run the analysis and verification first.")
            return
        
        print(f"✓ Loaded {len(comparison_results)} circuit results\n")
        
        # Visualize and save to algorithm directory
        algorithm_path = Path(algorithm_dir)
        algorithm_path.mkdir(parents=True, exist_ok=True)
        
        # Generate and save visualization (overwrites existing files)
        df = visualize_tvd_with_sequences(
            comparison_results, 
            sequence_analysis, 
            output_dir=str(algorithm_path)
        )
        
        # Print summary
        print_detailed_summary(df)
        
        # Save summary CSV to algorithm directory (overwrites existing)
        output_csv = algorithm_path / 'tvd_summary.csv'
        pd.DataFrame(df).to_csv(output_csv, index=False)
        print(f"\n✅ Summary saved to: {output_csv}")
        
        print("✅ Visualization completed (saved to algorithm directory, no window opened)")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


def run_full_pipeline(
    algorithm_dir: str,
    config_file: str = 'quantum_config.json',
    skip_initial_test: bool = False,
    tolerance: float = 0.01,
    max_granularity: int = 16,
    test_mode: bool = False,
    skip_verification: bool = False,
    skip_visualization: bool = False
):
    """
    Run the complete delta debugging pipeline.
    
    Args:
        algorithm_dir: Directory containing circuit files (e.g., 'artifacts/bv3-1')
        config_file: Path to quantum config file
        skip_initial_test: Skip the initial execution test
        tolerance: Tolerance threshold for delta debugging
        max_granularity: Maximum splitting depth
        test_mode: Whether to run in test mode
        skip_verification: Skip circuit verification step
        skip_visualization: Skip visualization step
    """
    print("="*60)
    print(f"🚀 Delta Debugging Pipeline: {algorithm_dir}")
    print("="*60)
    
    # Step 1: Load circuit
    artifacts_path = Path(algorithm_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = REPO_ROOT / artifacts_path
    if not artifacts_path.exists():
        print(f"❌ Directory not found: {algorithm_dir}")
        return
    
    isa = load_circuit(str(artifacts_path))
    if isa is None:
        return
    
    # Step 2: Initial test (optional)
    if not skip_initial_test:
        run_initial_test(isa, config_file)
    
    # Step 3: Delta debugging (reports saved with timestamps, never overwritten)
    debug_result = run_delta_debugging(
        isa,
        output_dir=str(artifacts_path),
        config_file=config_file,
        tolerance=tolerance,
        max_granularity=max_granularity,
        test_mode=test_mode
    )
    
    if debug_result is None:
        print("❌ Delta debugging failed, stopping pipeline")
        return
    
    # Step 4: Analyze segments (analyzes ALL historical reports, overwrites seq_XXX circuits)
    # This aggregates findings from all delta debug reports in the algorithm directory
    analysis = analyze_segments(str(artifacts_path), "tmp")
    
    if analysis is None:
        print("⚠️  Segment analysis failed, but continuing...")
    
    # Step 5: Verify circuits (overwrites circuit_comparison_results.json)
    if not skip_verification:
        verify_circuits_workflow(str(artifacts_path), config_file=config_file)
    
    # Step 6: Visualize results (overwrites tvd_analysis.png and tvd_summary.csv)
    if not skip_visualization:
        visualize_results_workflow(str(artifacts_path))
    
    print("\n" + "="*60)
    print("✅ Pipeline completed!")
    print("="*60)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Run delta debugging pipeline for quantum circuits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for bv3-1
  python run_delta_debug.py --algorithm artifacts/bv3-1
  
  # Run with custom tolerance
  python run_delta_debug.py --algorithm artifacts/qaoa3-1 --tolerance 0.02
  
  # Skip initial test and visualization
  python run_delta_debug.py --algorithm artifacts/qft4-1 --skip-initial-test --skip-visualization
        """
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        required=True,
        help='Algorithm directory (e.g., artifacts/bv3-1 or bv3-1)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='quantum_config.json',
        help='Path to quantum config file (default: quantum_config.json)'
    )
    
    parser.add_argument(
        '--skip-initial-test',
        action='store_true',
        help='Skip the initial execution test step'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.01,
        help='Tolerance threshold for delta debugging (default: 0.01)'
    )
    
    parser.add_argument(
        '--max-granularity',
        type=int,
        default=16,
        help='Maximum splitting depth (default: 16)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode (noisy vs ideal instead of real vs noisy)'
    )
    
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip circuit verification step'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization step'
    )
    
    args = parser.parse_args()
    
    # Normalize algorithm directory path
    algorithm_dir = args.algorithm
    if not algorithm_dir.startswith('artifacts/'):
        algorithm_dir = f'artifacts/{algorithm_dir}'
    
    run_full_pipeline(
        algorithm_dir=algorithm_dir,
        config_file=args.config,
        skip_initial_test=args.skip_initial_test,
        tolerance=args.tolerance,
        max_granularity=args.max_granularity,
        test_mode=args.test_mode,
        skip_verification=args.skip_verification,
        skip_visualization=args.skip_visualization
    )


if __name__ == '__main__':
    main()
