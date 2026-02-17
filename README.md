# Sequence Loss Analysis Guide

This guide explains the active analysis outputs for `artifacts/grover3-3`.

Current chart set:
- `slot/1.png`
- `slot/2.png`
- `slot/3.png`
- `slot/4.png`

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

### Core metrics
- `reports_caught`: number of report files where a sequence is caught at least once.
- `report_hit_rate = reports_caught / total_reports`
- `caught_occurrences`: number of problematic hits for a sequence.
- `total_occurrences`: number of all appearances for the same sequence.
- `occurrence_capture_rate = caught_occurrences / total_occurrences`

## Figure Guide

## 1) `slot/1.png`

### What it answers
Which sequences are repeatedly associated with problematic segments across multiple reports.

### Axes and labels
- X-axis: `report_hit_rate`
- Y-axis: sequence string
- Bar annotation: `reports_caught / total_reports`

### Interpretation
- Higher bars indicate stronger cross-report reproducibility.

## 2) `slot/2.png`

### What it answers
How risky a sequence is when it appears.

### Axes and labels
- X-axis: `occurrence_capture_rate`
- Y-axis: sequence string (top cohort selected by caught activity)
- Bar annotation: `caught_occurrences / total_occurrences`

### Interpretation
- Near `1.0`: sequence is frequently problematic when it appears.
- Lower values: sequence may be common but weakly associated with problem segments.

## 3) `slot/3.png`

### What it answers
Where in the circuit (operation index) problematic hits concentrate.

### Axes and labels
- X-axis: operation index (`1..N`)
- Y-axis: stacked problem-hit rate
- Colors: per-report contribution

### Interpretation
- Taller bars indicate positional hotspots.

## 4) `slot/4.png`

### What it answers
A concrete verification snapshot of expected vs observed error rate.

### Plot contents
- Bar 1: Expected error rate from repeated noisy-simulator runs.
- Bar 2: Observed error rate from real-backend referenced TVD.

### Interpretation
- If the observed bar is much higher than expected, the segment is likely under-modeled by the current noise assumptions.
- This figure is intended as a concise evidence slide for reporting a concrete deviation case.

## Recommended reading order

1. `1` for cross-report stability.
2. `2` for conditional risk.
3. `3` for location in the circuit.
4. `4` for a concrete verification-style summary.
