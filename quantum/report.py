"""
Delta Debug visual report generator.
Generates an HTML report from the DDMin result JSON.
"""

from typing import Dict, Any
import json


def generate_html_report(result: Dict[str, Any], output_path: str = None) -> str:
    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"report_{ts}.html"

    html = _generate_html_content(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report generated: {output_path}")
    return output_path


def _generate_html_content(result: Dict[str, Any]) -> str:
    meta = result['meta']
    problematic_segments = result['problematic_segments']
    ddmin_log = result['ddmin_log']
    segments_info = result['segments_info']

    total_segments = len(segments_info)
    baseline_entry = next(e for e in ddmin_log if e['action'] == 'baseline')
    baseline_loss = baseline_entry['loss']
    test_count = len(ddmin_log)

    eval_info = meta['evaluation']
    tolerance = eval_info['tolerance']

    summary = _section_summary(meta, total_segments, problematic_segments, baseline_loss, test_count, eval_info)
    segments = _section_problematic_segments(problematic_segments, segments_info)
    process = _section_ddmin_process(ddmin_log, tolerance)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Delta Debug Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif; background:#f0f2f5; padding:20px; color:#333; }}
.container {{ max-width:1200px; margin:0 auto; }}
.header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:30px; border-radius:12px 12px 0 0; text-align:center; }}
.header h1 {{ font-size:2em; margin-bottom:5px; }}
.header p {{ opacity:0.9; }}
.content {{ background:#fff; padding:30px; border-radius:0 0 12px 12px; }}
.section {{ margin-bottom:40px; }}
.section-title {{ font-size:1.4em; color:#667eea; margin-bottom:15px; padding-bottom:8px; border-bottom:2px solid #667eea; }}
.card {{ background:#f8f9fa; border-radius:8px; padding:20px; margin-bottom:15px; border:1px solid #e9ecef; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:15px; margin-bottom:20px; }}
.stat-card {{ background:#fff; padding:15px; border-radius:8px; text-align:center; border:1px solid #e9ecef; }}
.stat-card.highlight {{ border-color:#dc3545; background:#fff5f5; }}
.stat-label {{ font-size:0.85em; color:#6c757d; }}
.stat-value {{ font-size:1.6em; font-weight:bold; color:#667eea; }}
.config-grid {{ display:flex; flex-wrap:wrap; gap:15px 30px; }}
.config-item {{ font-size:0.9em; }} .config-item span {{ color:#6c757d; }} .config-item strong {{ color:#333; }}
.seg-box {{ background:#fff; border:2px solid #dc3545; border-radius:8px; padding:15px; margin-bottom:12px; }}
.seg-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #f0f0f0; }}
.seg-id {{ font-weight:bold; color:#dc3545; }}
.seg-desc {{ color:#6c757d; font-style:italic; font-size:0.9em; }}
.op-item {{ padding:6px 10px; margin:4px 0; background:#f8f9fa; border-radius:4px; font-family:monospace; font-size:0.85em; border-left:3px solid #667eea; }}
.badge {{ display:inline-block; padding:3px 8px; border-radius:10px; font-size:0.8em; font-weight:600; margin:2px; }}
.badge-danger {{ background:#f8d7da; color:#721c24; }}
.badge-info {{ background:#d1ecf1; color:#0c5460; }}
.badge-warning {{ background:#fff3cd; color:#856404; }}
.step {{ padding:12px 15px; margin-bottom:8px; background:#fff; border-radius:6px; border-left:4px solid #007bff; font-size:0.9em; }}
.step.progressed {{ background:#d4edda; border-left-color:#28a745; }}
.step.baseline {{ border-left-color:#6c757d; }}
.step-header {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:5px; margin-bottom:4px; }}
.step-label {{ font-weight:600; }}
.step-loss {{ color:#dc3545; font-weight:bold; }}
.step-delta {{ font-size:0.85em; }}
.step-detail {{ color:#6c757d; font-size:0.85em; }}
.footer {{ text-align:center; color:#6c757d; padding:15px; font-size:0.85em; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Delta Debug Report</h1>
    <p>Quantum circuit fault localization</p>
  </div>
  <div class="content">
    {summary}
    {segments}
    {process}
  </div>
  <div class="footer">Generated {meta.get('timestamp', 'N/A')}</div>
</div>
</body>
</html>"""

def _section_summary(meta, total_segments, problematic_segments, baseline_loss, test_count, eval_info):
    return f"""
    <div class="section">
      <h2 class="section-title">Summary</h2>
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-label">Total Segments</div><div class="stat-value">{total_segments}</div></div>
        <div class="stat-card highlight"><div class="stat-label">Problematic</div><div class="stat-value">{len(problematic_segments)}</div></div>
        <div class="stat-card"><div class="stat-label">Baseline Loss</div><div class="stat-value">{baseline_loss:.4f}</div></div>
        <div class="stat-card"><div class="stat-label">Tests Run</div><div class="stat-value">{test_count}</div></div>
      </div>
      <div class="card">
        <div class="config-grid">
          <div class="config-item"><span>Mode:</span> <strong>{eval_info.get('baseline_execution_type','N/A')} vs {eval_info.get('test_execution_type','N/A')}</strong></div>
          <div class="config-item"><span>Shots:</span> <strong>{eval_info['shots']}</strong></div>
          <div class="config-item"><span>Tolerance:</span> <strong>{eval_info.get('tolerance','N/A')}</strong></div>
          <div class="config-item"><span>Max Granularity:</span> <strong>{eval_info.get('max_granularity','N/A')}</strong></div>
          <div class="config-item"><span>Qubits:</span> <strong>{meta.get('circuit_info',{}).get('total_qubits','N/A')}</strong></div>
        </div>
      </div>
    </div>"""


def _section_problematic_segments(problematic_segments, segments_info):
    if not problematic_segments:
        return '<div class="section"><h2 class="section-title">Result</h2><div class="card" style="background:#d4edda;text-align:center;padding:30px;color:#155724;">No problematic segments detected.</div></div>'

    ops_count = {}
    affected_qubits = set()
    total_2q = 0
    segment_boxes = []

    for seg_idx in problematic_segments:
        if seg_idx >= len(segments_info):
            continue
        seg = segments_info[seg_idx]
        comp = seg.get('complexity', {})
        total_2q += comp.get('two_qubit_gates', 0)

        ops_html = []
        for inst in seg.get('instructions', []):
            op = inst['operation']
            ops_count[op] = ops_count.get(op, 0) + 1
            affected_qubits.update(inst['qubits'])
            qstr = ','.join(map(str, inst['qubits']))
            pstr = f" ({', '.join(f'{p:.4f}' if isinstance(p, float) else str(p) for p in inst['params'])})" if inst.get('params') else ''
            ops_html.append(f'<div class="op-item"><strong>{op}</strong>({qstr}){pstr}</div>')

        segment_boxes.append(f"""
        <div class="seg-box">
          <div class="seg-header"><span class="seg-id">Segment {seg['layer_id']}</span><span class="seg-desc">{seg['description']}</span></div>
          {''.join(ops_html)}
        </div>""")

    op_badges = ' '.join(f'<span class="badge badge-danger">{op}: {c}x</span>' for op, c in ops_count.items())
    q_badges = ' '.join(f'<span class="badge badge-info">Q{q}</span>' for q in sorted(affected_qubits))

    return f"""
    <div class="section">
      <h2 class="section-title">Problematic Segments</h2>
      <div class="card">
        <div style="margin-bottom:10px">{op_badges} <span class="badge badge-warning">2Q Gates: {total_2q}</span></div>
        <div>{q_badges}</div>
      </div>
      {''.join(segment_boxes)}
    </div>"""

def _section_ddmin_process(log_entries, tolerance):
    if not log_entries:
        return ""

    steps = []
    for i, e in enumerate(log_entries):
        action = e.get('action', 'unknown')
        loss = e.get('loss', 0)
        is_baseline = action in ('baseline',)
        progressed = e.get('progressed', False)

        cls = 'baseline' if is_baseline else ('progressed' if progressed else '')

        if is_baseline:
            label = 'BASELINE'
            detail = f"Full circuit ({len(e.get('kept', []))} segments)"
            delta_html = ''
        else:
            label = action.replace('_', ' ').upper()
            excluded = e.get('excluded', [])
            detail = f"Excluded {len(excluded)} segments: {excluded[:6]}{'...' if len(excluded) > 6 else ''}"
            delta_loss = e.get('delta_loss', 0)
            norm = e.get('normalized_score')
            dcolor = '#28a745' if delta_loss > 0 else ('#dc3545' if delta_loss < 0 else '#6c757d')
            delta_html = f'<span class="step-delta" style="color:{dcolor}">dL:{delta_loss:+.4f}</span>'
            if norm is not None:
                ncolor = '#28a745' if tolerance and norm > tolerance else '#6c757d'
                delta_html += f' <span class="step-delta" style="color:{ncolor}">score:{norm:.5f}</span>'
            if progressed:
                delta_html += ' <strong style="color:#28a745">NARROWED</strong>'

        steps.append(f"""
        <div class="step {cls}">
          <div class="step-header">
            <span class="step-label">#{i+1} {label}</span>
            <span><span class="step-loss">Loss: {loss:.4f}</span> {delta_html}</span>
          </div>
          <div class="step-detail">{detail}</div>
        </div>""")

    return f"""
    <div class="section">
      <h2 class="section-title">DDMin Process</h2>
      <div class="card">{''.join(steps)}</div>
    </div>"""


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python report.py <result_json_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        result = json.load(f)
    generate_html_report(result)
