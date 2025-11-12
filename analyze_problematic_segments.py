#!/usr/bin/env python3
"""
Analyze continuous operation sequences in problematic segments.
"""

import json
import sys
import glob
import os
from collections import defaultdict
from typing import List, Tuple

def load_report(filepath: str) -> dict:
    """Load JSON report"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_operations(segment: dict) -> List[Tuple[str, Tuple[int]]]:
    """Extract operations from segment: (operation, qubits)"""
    operations = []
    for op in segment.get('operations', []):
        op_name = op['operation']
        qubits = tuple(sorted(op['qubits']))
        operations.append((op_name, qubits))
    return operations

def format_op(op: Tuple[str, Tuple[int]]) -> str:
    """Format operation: 'rz(143)' or 'cz(136,143)'"""
    op_name, qubits = op
    qubits_str = ','.join(map(str, qubits))
    return f"{op_name}({qubits_str})"

def extract_sequences(operations: List[Tuple], min_len: int = 2, max_len: int = 4) -> List[Tuple]:
    """Extract continuous operation sequences"""
    sequences = []
    n = len(operations)
    
    for length in range(min_len, min(max_len + 1, n + 1)):
        for i in range(n - length + 1):
            seq = operations[i:i+length]
            op_seq = tuple([format_op(op) for op in seq])
            sequences.append(op_seq)
    
    return sequences

def analyze_sequences(reports: List[dict]) -> dict:
    """Analyze continuous operations in problematic segments"""
    
    all_seqs = defaultdict(lambda: {'count': 0, 'reports': set(), 'segments': []})
    
    for report_idx, report in enumerate(reports):
        prob_segs = report.get('problematic_segments', [])
        segments_info = {seg['layer_id']: seg for seg in report.get('analysis', {}).get('segments', [])}
        
        for seg_id in prob_segs:
            if seg_id in segments_info:
                ops = extract_operations(segments_info[seg_id])
                sequences = extract_sequences(ops, min_len=2, max_len=4)
                
                for op_seq in sequences:
                    all_seqs[op_seq]['count'] += 1
                    all_seqs[op_seq]['reports'].add(report_idx)
                    all_seqs[op_seq]['segments'].append({'report': report_idx, 'segment': seg_id})
    
    num_reports = len(reports)
    common_seqs = {}
    frequent_seqs = {}
    
    for seq_key, data in all_seqs.items():
        reports_list = sorted(list(data['reports']))
        if len(reports_list) == num_reports:
            common_seqs[seq_key] = {'count': data['count'], 'reports': reports_list}
        if data['count'] >= 3 or len(reports_list) >= 2:
            frequent_seqs[seq_key] = {'count': data['count'], 'reports': reports_list}
    
    common_seqs = dict(sorted(common_seqs.items(), key=lambda x: x[1]['count'], reverse=True))
    frequent_seqs = dict(sorted(frequent_seqs.items(), key=lambda x: (x[1]['count'], len(x[1]['reports'])), reverse=True))
    
    format_seq = lambda k: ' → '.join(k) if isinstance(k, tuple) else str(k)
    
    return {
        'num_reports': num_reports,
        'common_sequences': {format_seq(k): v for k, v in common_seqs.items()},
        'frequent_sequences': {format_seq(k): v for k, v in frequent_seqs.items()},
        'summary': {
            'total_unique_sequences': len(all_seqs),
            'sequences_in_all_reports': len(common_seqs),
            'frequent_sequences': len(frequent_seqs)
        }
    }

def print_analysis(analysis: dict):
    """Print analysis results"""
    print("=" * 80)
    print("Sequence Analysis Summary")
    print("=" * 80)
    
    summary = analysis['summary']
    print(f"\n📊 Reports analyzed: {analysis['num_reports']}")
    print(f"📊 Total unique sequences: {summary['total_unique_sequences']}")
    print(f"📊 Common sequences (all reports): {summary['sequences_in_all_reports']}")
    print(f"📊 Frequent sequences: {summary['frequent_sequences']}")
    
    # Common sequences
    if analysis['common_sequences']:
        print(f"\n{'='*80}")
        print("🔴 Common Sequences (appear in ALL reports)")
        print('='*80)
        for seq, data in list(analysis['common_sequences'].items())[:10]:
            print(f"  {seq}")
            print(f"    Count: {data['count']}, Reports: {data['reports']}")
    
    # Frequent sequences
    if analysis['frequent_sequences']:
        print(f"\n{'='*80}")
        print("⚠️  Frequent Sequences (≥3 occurrences or ≥2 reports)")
        print('='*80)
        for seq, data in list(analysis['frequent_sequences'].items())[:15]:
            print(f"  {seq}")
            print(f"    Count: {data['count']}, Reports: {data['reports']}")

def main():
    data_dir = "delta_debug_data"
    output_dir = "tmp"
    
    # Find report files
    if not os.path.exists(data_dir):
        print(f"❌ Directory '{data_dir}/' not found")
        sys.exit(1)
    
    report_files = sorted(glob.glob(os.path.join(data_dir, "delta_debug_report_*.json")))
    if not report_files:
        print(f"❌ No delta_debug_report_*.json files found in '{data_dir}/'")
        sys.exit(1)
    
    # Load reports
    print(f"Loading {len(report_files)} report(s) from {data_dir}/...")
    reports = []
    for fpath in report_files:
        try:
            reports.append(load_report(fpath))
        except Exception as e:
            print(f"  ⚠️  Skipped {fpath}: {e}")
    
    if not reports:
        print("❌ No reports loaded successfully")
        sys.exit(1)
    
    print(f"✓ Loaded {len(reports)} reports\n")
    
    # Analyze
    analysis = analyze_sequences(reports)
    print_analysis(analysis)
    
    # Save to tmp directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sequence_analysis.json")
    
    with open(output_path, 'w') as f:
        json.dump({
            'summary': analysis['summary'],
            'common_sequences': analysis['common_sequences'],
            'frequent_sequences': analysis['frequent_sequences']
        }, f, indent=2)
    
    print(f"\n✅ Analysis saved to: {output_path}")

if __name__ == "__main__":
    main()
