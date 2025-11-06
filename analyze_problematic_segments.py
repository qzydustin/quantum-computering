#!/usr/bin/env python3
"""
Analyze continuous operation sequences in problematic segments across multiple delta debug reports.
Identifies which continuous operation sequences frequently appear in problematic segments.
"""

import json
import sys
import glob
from collections import Counter, defaultdict
from typing import List, Tuple

def load_report(filepath: str) -> dict:
    """Load JSON report"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_operations(segment: dict) -> List[Tuple[str, Tuple[int], Tuple]]:
    """Extract operations from segment: (operation, qubits, params)"""
    operations = []
    for op in segment.get('operations', []):
        op_name = op['operation']
        qubits = tuple(sorted(op['qubits']))
        params = tuple(op.get('params', []))
        operations.append((op_name, qubits, params))
    return operations

def format_op_qubit(op: Tuple[str, Tuple[int], Tuple]) -> str:
    """Format operation with qubits: 'rz(143)' or 'cz(136,143)'"""
    op_name, qubits, params = op
    qubits_str = ','.join(map(str, qubits))
    return f"{op_name}({qubits_str})"

def extract_sequences(operations: List[Tuple], min_len: int = 2, max_len: int = 4) -> List[Tuple]:
    """Extract continuous operation sequences"""
    sequences = []
    n = len(operations)
    
    for length in range(min_len, min(max_len + 1, n + 1)):
        for i in range(n - length + 1):
            seq = operations[i:i+length]
            op_seq = tuple([format_op_qubit(op) for op in seq])
            full_seq = tuple(seq)
            sequences.append((op_seq, full_seq))
    
    return sequences

def analyze_sequences(reports: List[dict]) -> dict:
    """Analyze continuous operations in problematic segments"""
    
    all_seqs = defaultdict(lambda: {'count': 0, 'reports': [], 'segments': []})
    
    for report_idx, report in enumerate(reports):
        prob_segs = report.get('problematic_segments', [])
        segments_info = {seg['layer_id']: seg for seg in report.get('analysis', {}).get('segments', [])}
        
        for seg_id in prob_segs:
            if seg_id in segments_info:
                ops = extract_operations(segments_info[seg_id])
                sequences = extract_sequences(ops, min_len=2, max_len=4)
                
                for op_seq, full_seq in sequences:
                    all_seqs[op_seq]['count'] += 1
                    if report_idx not in all_seqs[op_seq]['reports']:
                        all_seqs[op_seq]['reports'].append(report_idx)
                    all_seqs[op_seq]['segments'].append({
                        'report': report_idx,
                        'segment_id': seg_id,
                        'full_sequence': full_seq
                    })
    
    num_reports = len(reports)
    common_seqs = {}
    frequent_seqs = {}
    
    for seq_key, data in all_seqs.items():
        if len(data['reports']) == num_reports:
            common_seqs[seq_key] = data
        if data['count'] >= 3 or len(data['reports']) >= 2:
            frequent_seqs[seq_key] = data
    
    common_seqs = dict(sorted(common_seqs.items(), key=lambda x: x[1]['count'], reverse=True))
    frequent_seqs = dict(sorted(frequent_seqs.items(), key=lambda x: (x[1]['count'], len(x[1]['reports'])), reverse=True))
    
    def format_key(seq_key):
        return ' → '.join(seq_key) if isinstance(seq_key, tuple) else str(seq_key)
    
    return {
        'num_reports': num_reports,
        'common_sequences': {format_key(k): v for k, v in common_seqs.items()},
        'frequent_sequences': {format_key(k): v for k, v in frequent_seqs.items()},
        'summary': {
            'total_unique_sequences': len(all_seqs),
            'sequences_in_all_reports': len(common_seqs),
            'frequent_sequences': len(frequent_seqs)
        }
    }

def print_analysis(analysis: dict, report_files: List[str]):
    """Print analysis results"""
    print("=" * 80)
    print("Continuous Operation Sequences Analysis - Problematic Segments")
    print("=" * 80)
    
    print(f"\nReports: {analysis['num_reports']}")
    for i, fname in enumerate(report_files):
        print(f"  [{i}] {fname}")
    
    print(f"\nSummary:")
    print(f"  Total unique sequences: {analysis['summary']['total_unique_sequences']}")
    print(f"  Sequences in all reports: {analysis['summary']['sequences_in_all_reports']}")
    print(f"  Frequent sequences: {analysis['summary']['frequent_sequences']}")
    
    # Common sequences
    print(f"\n" + "=" * 80)
    print("Common Sequences (in all reports)")
    print("=" * 80)
    
    if analysis['common_sequences']:
        for seq, data in list(analysis['common_sequences'].items())[:20]:
            print(f"\n{seq}")
            print(f"  Count: {data['count']} (in {len(data['reports'])} reports)")
            seg_display = ', '.join([f"report {s['report']} segment {s['segment_id']}" for s in data['segments'][:5]])
            print(f"  Segments: {seg_display}")
            if len(data['segments']) > 5:
                print(f"    ... {len(data['segments']) - 5} more")
    else:
        print("\nNo common sequences found")
    
    # Frequent sequences
    print(f"\n" + "=" * 80)
    print("Frequent Sequences (≥3 occurrences or ≥2 reports)")
    print("=" * 80)
    
    if analysis['frequent_sequences']:
        for seq, data in list(analysis['frequent_sequences'].items())[:30]:
            print(f"\n{seq}")
            print(f"  Count: {data['count']}, Reports: {data['reports']}")
            seg_display = ', '.join([f"report {s['report']} segment {s['segment_id']}" for s in data['segments'][:3]])
            print(f"  Segments: {seg_display}")
            if len(data['segments']) > 3:
                print(f"    ... {len(data['segments']) - 3} more")
    else:
        print("\nNo frequent sequences found")

def convert_for_json(obj):
    """Convert object for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, (set, frozenset)):
        return [convert_for_json(item) for item in obj]
    return obj

def main():
    if len(sys.argv) > 1:
        report_files = sys.argv[1:]
    else:
        report_files = sorted(glob.glob("delta_debug_report_*.json"))
        if not report_files:
            print("Error: No delta_debug_report_*.json files found")
            print("Usage: python analyze_problematic_segments.py [report1.json] [report2.json] ...")
            sys.exit(1)
    
    print(f"Found {len(report_files)} report file(s)")
    
    reports = []
    for fpath in report_files:
        try:
            reports.append(load_report(fpath))
            print(f"  ✓ {fpath}")
        except Exception as e:
            print(f"  ✗ {fpath}: {e}")
    
    if not reports:
        print("Error: No reports loaded")
        sys.exit(1)
    
    print(f"\nAnalyzing {len(reports)} report(s)...")
    analysis = analyze_sequences(reports)
    
    print_analysis(analysis, report_files)
    
    # Save key sequences
    output = {
        'summary': analysis['summary'],
        'common_sequences': analysis['common_sequences'],
        'frequent_sequences': analysis['frequent_sequences']
    }
    
    output_path = "key_continuous_sequences.json"
    with open(output_path, 'w') as f:
        json.dump(convert_for_json(output), f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    main()
