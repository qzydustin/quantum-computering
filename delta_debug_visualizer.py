"""
Delta Debug visual report generator.
Generates an HTML report showing the DDMin process and problematic segment analysis.
"""

from typing import Dict, Any, List
from pathlib import Path
import json


def generate_html_report(result: Dict[str, Any], output_path: str = None) -> str:
    """
    Generate an HTML visual report for Delta Debug.
    
    Args:
        result: The result dict returned by run_delta_debug_on_isa
        output_path: Output HTML file path; auto-generated if not provided
    
    Returns:
        The generated HTML file path
    """
    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"delta_debug_report_{ts}.html"
    
    html = _generate_html_content(result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML report generated: {output_path}")
    return output_path


def _generate_html_content(result: Dict[str, Any]) -> str:
    """Generate HTML content."""
    
    # Extract key data
    meta = result.get('meta', {})
    total_segments = result.get('total_segments', 0)
    problematic_segments = result.get('problematic_segments', [])
    baseline_loss = result.get('baseline_loss', 0)
    test_count = result.get('test_count', 0)
    ddmin_log = result.get('ddmin_log', [])
    evaluations = result.get('evaluations', [])
    segments_info = result.get('segments_info', [])
    analysis = result.get('analysis', {})
    
    # Generate sections
    summary_html = _generate_summary(meta, total_segments, problematic_segments, baseline_loss, test_count)
    timeline_html = _generate_timeline(evaluations, ddmin_log)
    problematic_segments_html = _generate_problematic_segments_analysis(
        problematic_segments, segments_info, analysis
    )
    ddmin_process_html = _generate_ddmin_process(ddmin_log)
    
    # assemble full HTML
    html = f"""
<!DOCTYPE html>
    <html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Delta Debug Visual Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .card-title {{
            font-size: 1.3em;
            color: #495057;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .metric {{
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 15px;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #6c757d;
            display: block;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .timeline {{
            position: relative;
            padding-left: 30px;
        }}
        
        .timeline::before {{
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(to bottom, #667eea, #764ba2);
        }}
        
        .timeline-item {{
            position: relative;
            padding: 15px 20px;
            background: white;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .timeline-item::before {{
            content: '';
            position: absolute;
            left: -32px;
            top: 20px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #667eea;
            border: 3px solid white;
            box-shadow: 0 0 0 3px #667eea;
        }}
        
        .timeline-item.baseline {{
            border-left-color: #28a745;
        }}
        
        .timeline-item.baseline::before {{
            background: #28a745;
            box-shadow: 0 0 0 3px #28a745;
        }}
        
        .timeline-item.ddmin {{
            border-left-color: #007bff;
        }}
        
        .timeline-item.ddmin::before {{
            background: #007bff;
            box-shadow: 0 0 0 3px #007bff;
        }}
        
        
        
        .timeline-item.progressed {{
            background: #d4edda;
            border-left-color: #28a745;
        }}
        
        .timeline-item.progressed::before {{
            background: #28a745;
            box-shadow: 0 0 0 3px #28a745;
        }}
        
        .timeline-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .timeline-mode {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .mode-baseline {{
            background: #28a745;
            color: white;
        }}
        
        .mode-ddmin {{
            background: #007bff;
            color: white;
        }}
        
        
        
        .timeline-loss {{
            font-size: 1.2em;
            font-weight: bold;
            color: #dc3545;
        }}
        
        .delta-badge {{
            font-size: 1.0em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .timeline-details {{
            color: #6c757d;
            font-size: 0.95em;
            line-height: 1.6;
        }}
        
        .segment-box {{
            background: white;
            border: 2px solid #dc3545;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }}
        
        .segment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f8f9fa;
        }}
        
        .segment-id {{
            font-size: 1.2em;
            font-weight: bold;
            color: #dc3545;
        }}
        
        .segment-desc {{
            color: #6c757d;
            font-style: italic;
        }}
        
        .operation-list {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
            margin-top: 10px;
        }}
        
        .operation-item {{
            padding: 8px;
            margin: 5px 0;
            background: white;
            border-radius: 4px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            border-left: 3px solid #667eea;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 2px solid #e9ecef;
        }}
        
        .stat-card.highlight {{
            border-color: #dc3545;
            background: #fff5f5;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 2px;
        }}
        
        .badge-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-danger {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .badge-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .progress-badge {{
            background: #28a745;
            color: white;
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        code {{
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                border-radius: 0;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .metric {{
                display: block;
                margin-right: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Delta Debug Visual Report</h1>
            <p>Quantum circuit fault localization and analysis</p>
        </div>
        
        <div class="content">
            {summary_html}
            {timeline_html}
            {problematic_segments_html}
            {ddmin_process_html}
        </div>
        
        <div class="footer">
            <p>Generated by Delta Debug Visualizer • {meta.get('timestamp', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def _generate_summary(meta: Dict, total_segments: int, problematic_segments: List[int], 
                     baseline_loss: float, test_count: int) -> str:
    """Generate summary section."""
    
    eval_info = meta.get('evaluation', {})
    
    html = f"""
    <div class="section">
        <h2 class="section-title">📊 Summary</h2>
        <div class="card">
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="metric-label">Total Segments</span>
                    <div class="metric-value">{total_segments}</div>
                </div>
                <div class="stat-card highlight">
                    <span class="metric-label">Problematic Segments</span>
                    <div class="metric-value">{len(problematic_segments)}</div>
                </div>
                <div class="stat-card">
                    <span class="metric-label">Baseline Loss</span>
                    <div class="metric-value">{baseline_loss:.4f}</div>
                </div>
                <div class="stat-card">
                    <span class="metric-label">Test Count</span>
                    <div class="metric-value">{test_count}</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">🔧 Configuration</div>
            <div class="metric">
                <span class="metric-label">Logical Qubits</span>
                <span class="metric-value" style="font-size: 1.3em;">{meta.get('logical_n_qubits', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Test Mode</span>
                <span class="metric-value" style="font-size: 1.1em;">{str(eval_info.get('test_mode', False))}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Baseline Exec</span>
                <span class="metric-value" style="font-size: 1.1em;">{eval_info.get('baseline_execution_type', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Test Exec</span>
                <span class="metric-value" style="font-size: 1.1em;">{eval_info.get('test_execution_type', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Tolerance Base</span>
                <span class="metric-value" style="font-size: 1.1em;">{eval_info.get('tolerance_base', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">DDMin Shots</span>
                <span class="metric-value" style="font-size: 1.1em;">{eval_info.get('shots_ddmin', 'N/A')}</span>
            </div>
        </div>
    </div>
    """
    
    return html


def _generate_timeline(evaluations: List[Dict], ddmin_log: List[Dict]) -> str:
    """Generate test timeline."""
    
    if not evaluations:
        return ""
    
    timeline_items = []
    
    for i, eval_item in enumerate(evaluations):
        mode = eval_item.get('mode', 'unknown')
        baseline = eval_item.get('baseline', 0)
        test = eval_item.get('test', 0)
        loss = eval_item.get('loss', 0)
        excluded = eval_item.get('excluded')
        included = eval_item.get('included')
        
        # Determine if this step progressed
        is_progressed = False
        if mode == 'ddmin' and ddmin_log:
            for log_entry in ddmin_log:
                if log_entry.get('excluded') == excluded and log_entry.get('progressed'):
                    is_progressed = True
                    break
        
        progress_badge = '<span class="progress-badge">✓ Progress</span>' if is_progressed else ''
        item_class = f"timeline-item {mode}" + (" progressed" if is_progressed else "")
        
        # Compute delta loss (prev_loss - current loss), if available
        delta_html = ''
        if mode == 'ddmin' and ddmin_log:
            matched_prev = None
            for log_entry in ddmin_log:
                if log_entry.get('excluded') == excluded:
                    matched_prev = log_entry.get('prev_loss')
                    break
            if matched_prev is not None:
                delta = matched_prev - loss
                color = '#28a745' if delta > 0 else ('#dc3545' if delta < 0 else '#6c757d')
                delta_html = f'<span class="delta-badge" style="color: {color};">ΔLoss: {delta:+.4f}</span>'
        
        # Build details
        if mode == 'baseline':
            details = f"Test full circuit (all {len(included) if included else 0} segments)"
        elif mode == 'ddmin':
            if excluded:
                details = f"Excluded segments: {excluded[:5]}{'...' if len(excluded) > 5 else ''} (total {len(excluded)})"
            else:
                details = "Test full circuit"
        else:
            details = "Unknown test"
        
        timeline_items.append(f"""
        <div class="{item_class}">
            <div class="timeline-header">
                <span class="timeline-mode mode-{mode}">#{i+1} {mode.upper()}</span>
                <span class="timeline-loss">Loss: {loss:.4f}</span>{delta_html}
                {progress_badge}
            </div>
            <div class="timeline-details">
                {details}<br>
                <small>Baseline: {baseline:.4f} | Test: {test:.4f}</small>
            </div>
        </div>
        """)
    
    html = f"""
    <div class="section">
        <h2 class="section-title">⏱️ Test Timeline</h2>
        <div class="card">
            <p style="color: #6c757d; margin-bottom: 20px;">
                Shows each test in chronological order. Green highlight indicates the step narrowed the failure region.
            </p>
            <div class="timeline">
                {''.join(timeline_items)}
            </div>
        </div>
    </div>
    """
    
    return html


def _generate_loss_curve(loss_scan: Dict) -> str:
    return ""


def _generate_problematic_segments_analysis(problematic_segments: List[int], 
                                            segments_info: List[Dict],
                                            analysis: Dict) -> str:
    """Generate problematic segments detailed analysis."""
    
    if not problematic_segments:
        return """
        <div class="section">
            <h2 class="section-title">✅ Problematic Segment Analysis</h2>
            <div class="card" style="background: #d4edda; border-color: #28a745;">
                <p style="color: #155724; font-size: 1.2em; text-align: center; padding: 20px;">
                    No obvious problematic segments detected. The circuit may be performing well overall.
                </p>
            </div>
        </div>
        """
    
    segment_boxes = []
    for seg_idx in problematic_segments:
        if seg_idx < len(segments_info):
            seg = segments_info[seg_idx]
            ops_html = []
            for inst in seg.get('instructions', []):
                qubits_str = ', '.join(map(str, inst['qubits']))
                params_str = ''
                if inst.get('params'):
                    params_str = f" <small>({inst['params']})</small>"
                ops_html.append(f"""
                <div class="operation-item">
                    <strong>{inst['operation']}</strong> → qubits[{qubits_str}]{params_str}
                </div>
                """)
            
            segment_boxes.append(f"""
            <div class="segment-box">
                <div class="segment-header">
                    <span class="segment-id">Layer {seg['layer_id']}</span>
                    <span class="segment-desc">{seg['description']}</span>
                </div>
                <div style="color: #6c757d; margin-bottom: 10px;">
                    Contains {len(seg.get('instructions', []))} operations
                </div>
                <div class="operation-list">
                    {''.join(ops_html)}
                </div>
            </div>
            """)
    
    # Operation statistics
    op_analysis = analysis.get('operation_analysis', {})
    op_badges = ' '.join([
        f'<span class="badge badge-danger">{op}: {count}×</span>'
        for op, count in op_analysis.items()
    ])
    
    # Qubit statistics
    qubit_analysis = analysis.get('qubit_analysis', {})
    affected_qubits = qubit_analysis.get('affected_qubits', [])
    qubit_badges = ' '.join([
        f'<span class="badge badge-info">Q{q}</span>'
        for q in affected_qubits
    ])
    
    html = f"""
    <div class="section">
        <h2 class="section-title">🔴 Problematic Segment Details</h2>
        
        <div class="card">
            <div class="card-title">📌 Quick Stats</div>
            <div class="metric">
                <span class="metric-label">Operations Involved</span>
                <div style="margin-top: 10px;">{op_badges}</div>
            </div>
            <div class="metric" style="margin-top: 20px;">
                <span class="metric-label">Qubits Involved</span>
                <div style="margin-top: 10px;">{qubit_badges}</div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">🔍 Segment Details</div>
            {''.join(segment_boxes)}
        </div>
    </div>
    """
    
    return html


def _generate_ddmin_process(ddmin_log: List[Dict]) -> str:
    """Generate DDMin process details."""
    
    if not ddmin_log:
        return ""
    
    steps_html = []
    for i, log_entry in enumerate(ddmin_log):
        action = log_entry.get('action', 'unknown')
        excluded = log_entry.get('excluded', [])
        loss = log_entry.get('loss', 0)
        prev_loss = log_entry.get('prev_loss', 0)
        progressed = log_entry.get('progressed', False)
        
        action_name = "Test subset" if action == "test_subset" else "Test complement"
        progress_icon = "✓" if progressed else "○"
        card_class = "progressed" if progressed else ""
        
        # Delta loss
        delta = (prev_loss if prev_loss is not None else 0) - (loss if loss is not None else 0)
        dcolor = '#28a745' if delta > 0 else ('#dc3545' if delta < 0 else '#6c757d')
        delta_html = f'<span class="delta-badge" style="color: {dcolor};">ΔLoss: {delta:+.4f}</span>'
        
        steps_html.append(f"""
        <div class="timeline-item ddmin {card_class}">
            <div class="timeline-header">
                <span>{progress_icon} Step {i+1}: {action_name}</span>
                <span class="timeline-loss">Loss: {loss:.4f}</span>{delta_html}
            </div>
            <div class="timeline-details">
                Excluded segments: {excluded[:8]}{'...' if len(excluded) > 8 else ''} (total {len(excluded)})<br>
                <small>Previous loss: {prev_loss:.4f} → Current loss: {loss:.4f}</small>
                {' <strong style="color: #28a745;">(narrowed)</strong>' if progressed else ''}
            </div>
        </div>
        """)
    
    html = f"""
    <div class="section">
        <h2 class="section-title">🔄 DDMin Iterations</h2>
        <div class="card">
            <p style="color: #6c757d; margin-bottom: 20px;">
                DDMin uses a divide-and-conquer strategy to progressively narrow the problematic segments. For each subset or complement test, if the loss drops, the candidate set is updated.
            </p>
            <div class="timeline">
                {''.join(steps_html)}
            </div>
        </div>
    </div>
    """
    
    return html


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python delta_debug_visualizer.py <result_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    with open(json_file, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    output_path = generate_html_report(result)
    print(f"Report generated: {output_path}")

