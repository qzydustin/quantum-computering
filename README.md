# Sequence Loss Analysis Guide (grover3-3)

This guide explains the sequence-loss charts exported for `artifacts/grover3-3`.

Current recommended chart set:
- `slot/1.png`
- `slot/2.png`
- `slot/3.png`
- `slot/05_tvd_loss_vs_report_support.png`

The former frequency-only chart (`02_caught_occurrences_top_sequences.png`) is intentionally excluded from the default analysis flow because its signal is largely covered by the normalized chart (03).

## Scope and Data Model

### Input
- Artifact directory: `artifacts/grover3-3`
- Reports: `delta_debug_report_*.json`

### Sequence definition
- Sequence length: 3 to 4 consecutive operations.
- Rendering format:
  - `operation(qubits[, params]) -> operation(...) -> ...`

### Aggregation sources
- `caught`: sequences extracted only from problematic segments.
- `total`: sequences extracted from full segmented circuit information.

### Metrics
- `reports_caught`: number of report files where a sequence is caught at least once.
- `report_hit_rate = reports_caught / total_reports`
- `caught_occurrences`: number of problematic hits for a sequence.
- `total_occurrences`: number of all appearances for the same sequence.
- `occurrence_capture_rate = caught_occurrences / total_occurrences`

## Figure-by-Figure Explanation

## 1) `slot/1.png`

### What it answers
Which sequences are repeatedly associated with problematic segments across multiple reports.

### Axes and labels
- X-axis: `report_hit_rate`
- Y-axis: sequence string
- Bar annotation: `reports_caught / total_reports`

### Interpretation
- Higher bars indicate stronger cross-report reproducibility.
- This is the best chart for selecting stable candidates for deeper debugging.

## 2) `slot/2.png`

### What it answers
How risky a sequence is when it appears.

### Axes and labels
- X-axis: `occurrence_capture_rate`
- Y-axis: sequence string (top cohort selected by caught activity)
- Bar annotation: `caught_occurrences / total_occurrences`

### Interpretation
- Near `1.0`: when this sequence appears, it is frequently in problematic regions.
- Lower values: sequence may be common but weakly associated with problems.
- Use this chart to avoid over-prioritizing high-frequency but low-specificity patterns.

## 3) `slot/3.png`

### What it answers
Where in the circuit (operation index) problematic hits concentrate.

### Axes and legend
- X-axis: operation index (`1..N`)
- Y-axis: stacked problem-hit rate
- Colors: per-report contribution

### Interpretation
- Taller stacked bars indicate positional hotspots.
- If multiple report colors pile up at similar indices, that location is a robust hotspot.

## 4) `slot/05_tvd_loss_vs_report_support.png`

### What it answers
Whether high-TVD candidate segments also have strong cross-report support.

### Axes and labels
- Bars (left axis): `tvd_loss` per candidate circuit/sequence
- Line (right axis): `report_support_rate = len(reports) / total_reports`
- Bar annotation: `count` from sequence analysis

### Interpretation
- High bar + high line point: strong candidate (large distribution gap and reproducible across reports).
- High bar + low line point: possibly severe but unstable/rare candidate.
- Low bar + high line point: frequent pattern with weaker distribution deviation.

## Recommended Reading Order

1. Start with `01` to find sequences with strong cross-report consistency.
2. Use `03` to prioritize sequences with high conditional risk.
3. Use `04` to map those sequence risks back to concrete circuit positions.
4. Use `05` to validate that high-risk candidates also carry meaningful TVD evidence.

## Why `02` Was Removed From the Default Set

`02_caught_occurrences_top_sequences.png` is raw volume only.
- It is useful for quick counting.
- But it does not normalize by background occurrence.
- In practice, `03` carries the more decision-relevant signal for triage.

So the default minimal and high-value set is:
- `01` (stability across reports)
- `03` (risk density)
- `04` (positional localization)
- `05` (independent TVD evidence + report support)

## Reproducibility

Main notebook:
- `notebooks/sequence_loss_analysis.ipynb`

Export helper script:
- `scripts/export_sequence_loss_slot.py`

Run (venv):
- `.venv/bin/python scripts/export_sequence_loss_slot.py`

If matplotlib cache permissions are restricted:
- `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp HOME=/tmp .venv/bin/python scripts/export_sequence_loss_slot.py`
