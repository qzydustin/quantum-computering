#!/usr/bin/env python3
"""
Delta Debugging Experiment CLI.

Usage:
    python -m quantum.cli --algorithm artifacts/grover3-2
    python -m quantum.cli --algorithm artifacts/grover3-2 --test-mode
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from qiskit import qpy

from .delta_debug import run_delta_debug_on_isa
from .executor import QuantumExecutor
from .report import generate_html_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_circuit(artifacts_dir: Path):
    """Load the latest QPY circuit from an artifacts directory."""
    qpy_files = sorted(artifacts_dir.glob("*.qpy"), reverse=True)
    if not qpy_files:
        print(f"No .qpy files found in {artifacts_dir}")
        return None
    path = qpy_files[0]
    print(f"Loading circuit: {path}")
    with open(path, "rb") as f:
        return list(qpy.load(f))[0]


def run_experiment(
    algorithm_dir: str,
    config_file: str = "quantum_config.json",
    tolerance: float | None = None,
    max_granularity: int = 16,
    test_mode: bool = False,
    resume_from: str | None = None,
):
    artifacts_path = Path(algorithm_dir)
    if not artifacts_path.is_absolute():
        artifacts_path = REPO_ROOT / artifacts_path
    if not artifacts_path.exists():
        print(f"Directory not found: {artifacts_path}")
        return

    isa = load_circuit(artifacts_path)
    if isa is None:
        return

    config_path = str((REPO_ROOT / config_file).resolve())
    executor = QuantumExecutor(config_file=config_path)
    tolerance = (
        executor.config.get("execution", {})
        .get("delta_debug", {})
        .get("tolerance", 0.01)
        if tolerance is None
        else tolerance
    )

    # Load resume candidates from previous result
    resume_candidates = None
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_from
        with open(resume_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        resume_candidates = prev["problematic_segments"]
        print(f"Resuming from {resume_path}")
        print(f"  Previous problematic segments: {resume_candidates}")

    print("\n" + "=" * 60)
    print("Delta Debugging")
    print("=" * 60)

    # Prepare output path early for incremental saves
    artifacts_path.mkdir(parents=True, exist_ok=True)
    backend_name = executor.backend.name if executor.backend else "unknown"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{backend_name}_{ts}"
    json_path = artifacts_path / f"{base}.json"

    # When resuming, prepend previous log to incremental saves
    prev_log = prev.get("ddmin_log", []) if resume_from else []
    progress_step = 0

    def _fmt_metric(value, digits=4, signed=False):
        if isinstance(value, (int, float)):
            return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
        return "N/A"

    print("Progress: running DDMin steps...", flush=True)

    def on_step(result):
        nonlocal progress_step

        if prev_log:
            new_steps = [e for e in result["ddmin_log"] if e.get("action") != "baseline"]
            result = {**result, "ddmin_log": prev_log + new_steps}

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        if not result["ddmin_log"]:
            return

        entry = result["ddmin_log"][-1]
        if entry.get("action") == "baseline":
            return

        progress_step += 1
        action = entry.get("action", "step")
        excluded = len(entry.get("excluded", []))
        remaining = len(result.get("problematic_segments", []))
        loss_text = _fmt_metric(entry.get("loss"), digits=4)
        delta_text = _fmt_metric(entry.get("delta_loss"), digits=4, signed=True)
        score_text = _fmt_metric(entry.get("normalized_score"), digits=5, signed=True)
        status = "narrowed" if entry.get("progressed") else "searching"

        print(
            f"[Step {progress_step:03d}] {action} | exclude={excluded} | "
            f"loss={loss_text} | delta={delta_text} | score={score_text} | "
            f"remaining={remaining} | {status}",
            flush=True,
        )

    result = run_delta_debug_on_isa(
        executor=executor,
        isa_circuit=isa,
        tolerance=tolerance,
        max_granularity=max_granularity,
        test_mode=test_mode,
        resume_candidates=resume_candidates,
        on_step=on_step,
    )

    # Merge previous log for final result
    if prev_log:
        new_steps = [e for e in result["ddmin_log"] if e.get("action") != "baseline"]
        result["ddmin_log"] = prev_log + new_steps

    # Final save
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"JSON report: {json_path}")

    html_path = str(artifacts_path / f"{base}.html")
    generate_html_report(result, output_path=html_path)

    # Summary
    baseline = next(e for e in result["ddmin_log"] if e["action"] == "baseline")
    print(f"\nKey Findings:")
    print(f"  Total segments: {len(result['segments_info'])}")
    print(f"  Problematic: {result['problematic_segments']}")
    print(f"  Baseline loss: {baseline['loss']:.4f}")
    print(f"  Tests run: {len(result['ddmin_log'])}")


def main():
    parser = argparse.ArgumentParser(description="Delta debugging experiment runner")
    parser.add_argument("--algorithm", "-a", required=True, help="Artifacts directory")
    parser.add_argument("--config", default="quantum_config.json")
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--max-granularity", type=int, default=16)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--resume", help="Path to previous JSON result to resume from")
    args = parser.parse_args()

    algorithm_dir = args.algorithm
    if not algorithm_dir.startswith("artifacts/"):
        algorithm_dir = f"artifacts/{algorithm_dir}"

    run_experiment(
        algorithm_dir=algorithm_dir,
        config_file=args.config,
        tolerance=args.tolerance,
        max_granularity=args.max_granularity,
        test_mode=args.test_mode,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
