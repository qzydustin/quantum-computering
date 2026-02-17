from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, qpy

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum.metrics import calculate_tvd
from quantum.quantum_executor import QuantumExecutor

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "grover3-3"
OUT_DIR = REPO_ROOT / "slot"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_reports() -> list[dict]:
    files = sorted(glob.glob(str(ARTIFACT_DIR / "delta_debug_report_*.json")))
    out = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["__file"] = fp
        out.append(d)
    return out


def build_problem_hit_counter(reports: list[dict]) -> tuple[Counter, int]:
    hit_counter = Counter()
    max_op_index = -1
    for report in reports:
        problematic_ids = set(report.get("problematic_segments", []))
        full_segments = report.get("segments_info", [])
        seg_map = {s.get("layer_id"): s for s in full_segments if isinstance(s, dict)}

        for seg in full_segments:
            for inst in seg.get("instructions", []) if isinstance(seg, dict) else []:
                idx = inst.get("index") if isinstance(inst, dict) else None
                if isinstance(idx, int) and idx > max_op_index:
                    max_op_index = idx

        for seg_id in problematic_ids:
            seg = seg_map.get(seg_id)
            if not seg:
                continue
            for inst in seg.get("instructions", []):
                idx = inst.get("index") if isinstance(inst, dict) else None
                if isinstance(idx, int):
                    hit_counter[idx] += 1

    return hit_counter, max_op_index


def find_peak_regions(hit_counter: Counter, max_op_index: int) -> list[tuple[int, int, int, int]]:
    vals = [hit_counter.get(i, 0) for i in range(max_op_index + 1)]
    regions: list[tuple[int, int, int, int]] = []
    i = 0
    while i < len(vals):
        if vals[i] == 0:
            i += 1
            continue
        j = i
        max_hit = vals[i]
        total = 0
        while j < len(vals) and vals[j] > 0:
            max_hit = max(max_hit, vals[j])
            total += vals[j]
            j += 1
        regions.append((i, j - 1, max_hit, total))
        i = j

    # Sort by max height then area
    regions.sort(key=lambda x: (x[2], x[3]), reverse=True)
    return regions


def choose_representative_index(region: tuple[int, int, int, int], hit_counter: Counter) -> int:
    start, end, max_hit, _ = region
    candidates = [i for i in range(start, end + 1) if hit_counter.get(i, 0) == max_hit]
    return candidates[len(candidates) // 2]


def load_base_circuit() -> QuantumCircuit:
    qpy_files = sorted(ARTIFACT_DIR.glob("*.qpy"), reverse=True)
    if not qpy_files:
        raise RuntimeError(f"No .qpy found under {ARTIFACT_DIR}")
    with open(qpy_files[0], "rb") as f:
        circ = list(qpy.load(f))[0]
    return circ


def extract_local_subcircuit(base: QuantumCircuit, center_idx: int, radius: int = 1) -> tuple[QuantumCircuit, int, int, list[int]]:
    start = max(0, center_idx - radius)
    end = min(len(base.data) - 1, center_idx + radius)

    used_qubits: list[int] = []
    for i in range(start, end + 1):
        inst = base.data[i]
        for q in inst.qubits:
            q_idx = base.find_bit(q).index
            if q_idx not in used_qubits:
                used_qubits.append(q_idx)

    if not used_qubits:
        raise RuntimeError(f"No qubits used for instruction window [{start}, {end}]")

    sub = QuantumCircuit(base.num_qubits, len(used_qubits))

    for i in range(start, end + 1):
        inst = base.data[i]
        qidx = [base.find_bit(q).index for q in inst.qubits]
        cidx = [base.find_bit(c).index for c in inst.clbits]
        sub.append(inst.operation, qidx, cidx)

    for c_i, q_i in enumerate(used_qubits):
        sub.measure(q_i, c_i)

    return sub, start, end, used_qubits


def run_tvd_for_circuit(qe: QuantumExecutor, circuit: QuantumCircuit) -> float:
    noisy = qe.run_circuit(circuit, execution_type="noisy_simulator")
    real = qe.run_circuit(circuit, execution_type="real_device")
    tvd_loss, _ = calculate_tvd(noisy.get("counts", {}), real.get("counts", {}))
    return float(tvd_loss)


def save_plot(rows: list[dict], out_png: Path):
    labels = [r["label"] for r in rows]
    tvd_vals = [r["tvd_loss"] for r in rows]

    x = list(range(len(rows)))
    bars = plt.bar(x, tvd_vals, width=0.6, color="tomato", label="TVD loss (noisy vs real)")

    plt.xticks(x, labels)
    plt.ylabel("Loss")
    plt.title("Top-2 Peak Segments: TVD loss (real vs noisy)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    for b, v in zip(bars, tvd_vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_csv(rows: list[dict], out_csv: Path):
    fields = [
        "peak_rank",
        "center_op_index_1based",
        "window_start_1based",
        "window_end_1based",
        "region_start_1based",
        "region_end_1based",
        "region_max_hits",
        "region_total_hits",
        "used_qubits",
        "tvd_loss",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def main():
    reports = load_reports()
    if not reports:
        raise RuntimeError(f"No reports in {ARTIFACT_DIR}")

    hit_counter, max_idx = build_problem_hit_counter(reports)
    regions = find_peak_regions(hit_counter, max_idx)
    if len(regions) < 2:
        raise RuntimeError(f"Need at least 2 non-zero peak regions, found {len(regions)}")

    top2 = regions[:2]
    base = load_base_circuit()
    qe = QuantumExecutor(config_file=str(REPO_ROOT / "quantum_config.json"))

    rows = []
    for rank, region in enumerate(top2, start=1):
        center = choose_representative_index(region, hit_counter)
        sub, win_s, win_e, used_qubits = extract_local_subcircuit(base, center_idx=center, radius=1)
        tvd_loss = run_tvd_for_circuit(qe, sub)

        row = {
            "peak_rank": rank,
            "center_op_index_1based": center + 1,
            "window_start_1based": win_s + 1,
            "window_end_1based": win_e + 1,
            "region_start_1based": region[0] + 1,
            "region_end_1based": region[1] + 1,
            "region_max_hits": region[2],
            "region_total_hits": region[3],
            "used_qubits": str(used_qubits),
            "tvd_loss": tvd_loss,
            "label": f"peak{rank}\\nop#{center + 1}",
        }
        rows.append(row)

    out_png = OUT_DIR / "06_top2_peaks_real_vs_noisy_tvd.png"
    out_csv = OUT_DIR / "06_top2_peaks_real_vs_noisy_tvd.csv"
    save_plot(rows, out_png)
    save_csv(rows, out_csv)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")
    for r in rows:
        print(
            f"peak{r['peak_rank']} center_op={r['center_op_index_1based']} "
            f"window=[{r['window_start_1based']},{r['window_end_1based']}] "
            f"TVD={r['tvd_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
