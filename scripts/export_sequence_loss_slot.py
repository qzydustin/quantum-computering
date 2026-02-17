from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path.cwd().resolve()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "grover3-3"
OUT_DIR = REPO_ROOT / "slot"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def format_op(op: dict) -> str:
    op_name = op.get("operation", "?")
    qubits = op.get("qubits", [])
    params = op.get("params", [])
    q_str = ",".join(str(q) for q in qubits)
    if params:
        p_str = ", " + ", ".join(f"{float(p):.4f}" for p in params)
    else:
        p_str = ""
    return f"{op_name}({q_str}{p_str})"


def seq_key(ops: list[dict]) -> str:
    return " -> ".join(format_op(op) for op in ops)


def extract_sequences_from_ops(ops: list[dict], min_len: int = 3, max_len: int = 4) -> list[str]:
    out = []
    n = len(ops)
    for length in range(min_len, min(max_len + 1, n + 1)):
        for i in range(n - length + 1):
            out.append(seq_key(ops[i : i + length]))
    return out


def load_reports(artifact_dir: Path) -> list[dict]:
    files = sorted(artifact_dir.glob("delta_debug_report_*.json"))
    reports = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            d = json.load(f)
        d["__file"] = str(fp)
        reports.append(d)
    return reports


def _segment_ops(seg: dict) -> list[dict]:
    ops = seg.get("operations")
    if isinstance(ops, list) and ops:
        return ops
    inst = seg.get("instructions")
    if isinstance(inst, list) and inst:
        return [
            {
                "operation": i.get("operation"),
                "qubits": i.get("qubits", []),
                "params": i.get("params", []),
            }
            for i in inst
            if isinstance(i, dict) and i.get("operation")
        ]
    return []


def build_sequence_df(reports: list[dict], artifact_name: str, source: str) -> pd.DataFrame:
    rows = []
    for ridx, report in enumerate(reports):
        analysis = report.get("analysis") or {}

        if source == "caught":
            all_segments = analysis.get("segments", [])
            target_ids = set(report.get("problematic_segments", []))
            segments = [s for s in all_segments if s.get("layer_id") in target_ids]
        elif source == "total":
            segments = report.get("segments_info", [])
        else:
            raise ValueError(f"Unknown source: {source}")

        for seg in segments:
            ops = _segment_ops(seg)
            for s in extract_sequences_from_ops(ops, min_len=3, max_len=4):
                rows.append(
                    {
                        "artifact": artifact_name,
                        "report_idx": ridx,
                        "report_file": report.get("__file"),
                        "segment_id": seg.get("layer_id"),
                        "sequence": s,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["artifact", "report_idx", "report_file", "segment_id", "sequence"])
    return pd.DataFrame(rows)


def aggregate_sequence_stats(caught_df: pd.DataFrame, total_df: pd.DataFrame, total_reports: int) -> pd.DataFrame:
    if total_df.empty:
        return pd.DataFrame(
            columns=[
                "sequence",
                "caught_occurrences",
                "total_occurrences",
                "occurrence_capture_rate",
                "reports_caught",
                "report_hit_rate",
            ]
        )

    total_agg = total_df.groupby("sequence", as_index=False).agg(total_occurrences=("sequence", "size"))

    if caught_df.empty:
        caught_agg = pd.DataFrame(columns=["sequence", "caught_occurrences", "reports_caught"])
    else:
        caught_agg = caught_df.groupby("sequence", as_index=False).agg(
            caught_occurrences=("sequence", "size"),
            reports_caught=("report_file", lambda s: len(set(s))),
        )

    agg = total_agg.merge(caught_agg, on="sequence", how="left").fillna(
        {
            "caught_occurrences": 0,
            "reports_caught": 0,
        }
    )
    agg["caught_occurrences"] = agg["caught_occurrences"].astype(int)
    agg["reports_caught"] = agg["reports_caught"].astype(int)

    agg["occurrence_capture_rate"] = agg.apply(
        lambda r: (r["caught_occurrences"] / r["total_occurrences"]) if r["total_occurrences"] > 0 else 0.0,
        axis=1,
    )
    agg["report_hit_rate"] = agg["reports_caught"] / max(total_reports, 1)

    return agg.sort_values(["report_hit_rate", "occurrence_capture_rate", "caught_occurrences"], ascending=False)


def save_top_sequences_by_report_hit_rate(agg_df: pd.DataFrame, total_reports: int, out_path: Path, top_n: int = 15):
    top = agg_df.sort_values("report_hit_rate", ascending=False).head(top_n).copy().iloc[::-1]
    plt.figure(figsize=(12, max(6, top_n * 0.35)))
    bars = plt.barh(top["sequence"], top["report_hit_rate"], color="teal")
    for bar, num in zip(bars, top["reports_caught"]):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{int(num)}/{int(total_reports)}", va="center", fontsize=9)
    plt.xlabel("Report hit rate (reports_caught / total_reports)")
    plt.title("grover3-3: report hit rate for complex sequences (>=3 ops)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_top_sequences_by_caught_occurrences(agg_df: pd.DataFrame, out_path: Path, top_n: int = 15):
    top = agg_df.sort_values("caught_occurrences", ascending=False).head(top_n).copy().iloc[::-1]
    plt.figure(figsize=(12, max(6, top_n * 0.35)))
    plt.barh(top["sequence"], top["caught_occurrences"], color="steelblue")
    plt.xlabel("Caught occurrences")
    plt.title("grover3-3: caught occurrences for complex sequences (>=3 ops)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_top_sequences_by_occurrence_capture_rate(agg_df: pd.DataFrame, out_path: Path, top_n: int = 15):
    top = agg_df.sort_values("caught_occurrences", ascending=False).head(top_n).copy().iloc[::-1]
    plt.figure(figsize=(12, max(6, top_n * 0.35)))
    bars = plt.barh(top["sequence"], top["occurrence_capture_rate"], color="seagreen")
    for bar, num, den in zip(bars, top["caught_occurrences"], top["total_occurrences"]):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{int(num)}/{int(den)}", va="center", fontsize=9)
    plt.xlabel("Occurrence capture rate (caught_occurrences / total_occurrences)")
    plt.title("grover3-3: occurrence capture rate for top caught sequences (>=3 ops)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_operation_hit_series(reports: list[dict], total_reports: int) -> pd.DataFrame:
    hit_counter = Counter()
    max_op_index = -1
    per_report_hits = []

    for report in reports:
        report_counter = Counter()
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
                    report_counter[idx] += 1

        per_report_hits.append(report_counter)

    n_ops = max_op_index + 1 if max_op_index >= 0 else 0
    rows = []
    for zero_idx in range(n_ops):
        hits = hit_counter.get(zero_idx, 0)
        row = {
            "op_index": zero_idx + 1,
            "problem_hits": hits,
            "problem_hit_rate": hits / max(total_reports, 1),
        }
        for ridx, report_counter in enumerate(per_report_hits, 1):
            row[f"report_{ridx}"] = report_counter.get(zero_idx, 0) / max(total_reports, 1)
        rows.append(row)

    return pd.DataFrame(rows)


def save_operation_hit_series(op_df: pd.DataFrame, out_path: Path):
    max_x = int(op_df["op_index"].max())
    step = max(1, max_x // 20)
    report_cols = [c for c in op_df.columns if c.startswith("report_")]

    plt.figure(figsize=(14, 5))
    bottom = [0.0] * len(op_df)
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(report_cols):
        vals = op_df[col].tolist()
        plt.bar(op_df["op_index"], vals, bottom=bottom, color=cmap(i % 10), width=0.9, label=col)
        bottom = [b + v for b, v in zip(bottom, vals)]

    plt.xlim(1, max_x)
    plt.xticks(list(range(1, max_x + 1, step)))
    plt.xlabel("Operation Index in Circuit (1..N)")
    plt.ylabel("Problem Hit Rate (stacked by report)")
    plt.title("grover3-3: problem hits by operation index (1..N)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def load_tvd_summary(artifact_dir: Path) -> pd.DataFrame:
    tvd_path = artifact_dir / "tvd_summary.csv"
    if not tvd_path.exists():
        raise RuntimeError(f"Missing TVD summary: {tvd_path}")
    df = pd.read_csv(tvd_path)
    required = {"circuit", "sequence", "tvd_loss", "count", "reports"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"tvd_summary.csv missing columns: {sorted(missing)}")
    return df


def _parse_reports(value: object) -> list[int]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, int)]
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, int)]


def save_tvd_vs_report_support(tvd_df: pd.DataFrame, total_reports: int, out_path: Path):
    df = tvd_df.copy()
    df["tvd_loss"] = pd.to_numeric(df["tvd_loss"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    df["reports_len"] = df["reports"].apply(lambda v: len(_parse_reports(v)))
    df["report_support_rate"] = df["reports_len"] / max(total_reports, 1)
    df = df.sort_values("tvd_loss", ascending=False).reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    x = list(range(len(df)))
    bars = ax1.bar(x, df["tvd_loss"], color="tomato", alpha=0.8, label="TVD loss")
    ax1.set_ylabel("TVD loss")
    ax1.set_xlabel("Circuit (sorted by TVD loss)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["circuit"], rotation=40, ha="right")

    for bar, cnt in zip(bars, df["count"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"count={int(cnt)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()
    ax2.plot(x, df["report_support_rate"], color="navy", marker="o", linewidth=2, label="report support rate")
    ax2.set_ylabel("Report support rate")
    ax2.set_ylim(0, 1.05)

    ax1.grid(axis="y", alpha=0.25, linestyle="--")
    ax1.set_title("grover3-3: TVD loss with cross-report support")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    reports = load_reports(ARTIFACT_DIR)
    total_reports = len(reports)
    if total_reports == 0:
        raise RuntimeError(f"No delta debug reports found in: {ARTIFACT_DIR}")

    caught_seq_df = build_sequence_df(reports, ARTIFACT_DIR.name, source="caught")
    total_seq_df = build_sequence_df(reports, ARTIFACT_DIR.name, source="total")
    agg = aggregate_sequence_stats(caught_seq_df, total_seq_df, total_reports=total_reports)

    save_top_sequences_by_report_hit_rate(agg, total_reports, OUT_DIR / "01_report_hit_rate_top_sequences.png")
    save_top_sequences_by_occurrence_capture_rate(agg, OUT_DIR / "03_occurrence_capture_rate_top_sequences.png")

    op_df = build_operation_hit_series(reports, total_reports=total_reports)
    if op_df.empty:
        raise RuntimeError("Operation-hit series is empty; cannot render operation index chart.")
    save_operation_hit_series(op_df, OUT_DIR / "04_problem_hits_by_operation_index.png")
    tvd_df = load_tvd_summary(ARTIFACT_DIR)
    save_tvd_vs_report_support(tvd_df, total_reports, OUT_DIR / "05_tvd_loss_vs_report_support.png")

    print(f"Output directory: {OUT_DIR}")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"- {p.name}")


if __name__ == "__main__":
    main()
