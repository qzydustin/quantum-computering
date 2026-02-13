#!/usr/bin/env python3
"""
Analyze continuous operation sequences in problematic segments and generate circuits.
"""

import json
import glob
import os
import shutil
from collections import defaultdict
from typing import List, Tuple, Dict, Any
from qiskit import QuantumCircuit, qpy
import numpy as np

def load_report(filepath: str) -> dict:
    """Load JSON report"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_operations(segment: dict) -> List[Dict[str, Any]]:
    """Extract operations from segment with full details"""
    operations = []
    for op in segment.get('operations', []):
        op_dict = {
            'operation': op['operation'],
            'qubits': tuple(op['qubits']),
            'params': tuple(op.get('params', []))
        }
        operations.append(op_dict)
    return operations

def format_op(op: Dict[str, Any]) -> str:
    """Format operation: 'rz(143, 1.571)' or 'cz(136,143)'"""
    op_name = op['operation']
    qubits = op['qubits']
    params = op['params']
    
    qubits_str = ','.join(map(str, qubits))
    if params:
        params_str = ', ' + ', '.join([f"{p:.4f}" for p in params])
    else:
        params_str = ''
    return f"{op_name}({qubits_str}{params_str})"

def ops_match(op1: Dict[str, Any], op2: Dict[str, Any]) -> bool:
    """Check if two operations match exactly (gate, qubits, params)"""
    if op1['operation'] != op2['operation']:
        return False
    if op1['qubits'] != op2['qubits']:
        return False
    # Compare params with tolerance
    if len(op1['params']) != len(op2['params']):
        return False
    for p1, p2 in zip(op1['params'], op2['params']):
        if not np.isclose(p1, p2, rtol=1e-9, atol=1e-12):
            return False
    return True

def extract_sequences(operations: List[Dict], min_len: int = 2, max_len: int = 4) -> List[Tuple]:
    """Extract continuous operation sequences with position info"""
    sequences = []
    n = len(operations)
    
    for length in range(min_len, min(max_len + 1, n + 1)):
        for i in range(n - length + 1):
            seq = operations[i:i+length]
            # Store both formatted string and actual operations
            sequences.append((seq, i, i+length))
    
    return sequences

def sequence_to_key(seq: List[Dict[str, Any]]) -> str:
    """Convert sequence to a unique key string"""
    return ' → '.join([format_op(op) for op in seq])

def sequences_match(seq1: List[Dict], seq2: List[Dict]) -> bool:
    """Check if two sequences match exactly"""
    if len(seq1) != len(seq2):
        return False
    return all(ops_match(op1, op2) for op1, op2 in zip(seq1, seq2))

def analyze_sequences(reports: List[dict]) -> dict:
    """Analyze continuous operations in problematic segments"""
    
    # Store sequences with their actual operations
    all_seqs = defaultdict(lambda: {'count': 0, 'reports': set(), 'segments': [], 'example_ops': None})
    
    for report_idx, report in enumerate(reports):
        prob_segs = report.get('problematic_segments', [])
        segments_info = {seg['layer_id']: seg for seg in report.get('analysis', {}).get('segments', [])}
        
        for seg_id in prob_segs:
            if seg_id in segments_info:
                ops = extract_operations(segments_info[seg_id])
                sequences = extract_sequences(ops, min_len=2, max_len=4)
                
                for seq_ops, start_idx, end_idx in sequences:
                    seq_key = sequence_to_key(seq_ops)
                    all_seqs[seq_key]['count'] += 1
                    all_seqs[seq_key]['reports'].add(report_idx)
                    all_seqs[seq_key]['segments'].append({
                        'report': report_idx, 
                        'segment': seg_id,
                        'start': start_idx,
                        'end': end_idx
                    })
                    # Store example operations if not already stored
                    if all_seqs[seq_key]['example_ops'] is None:
                        all_seqs[seq_key]['example_ops'] = seq_ops
    
    num_reports = len(reports)
    all_sequences = {}
    
    for seq_key, data in all_seqs.items():
        reports_list = sorted(list(data['reports']))
        seq_data = {
            'count': data['count'], 
            'reports': reports_list,
            'operations': data['example_ops']
        }
        all_sequences[seq_key] = seq_data
    
    # Sort by count and number of reports
    all_sequences = dict(sorted(all_sequences.items(), key=lambda x: (x[1]['count'], len(x[1]['reports'])), reverse=True))
    
    return {
        'num_reports': num_reports,
        'all_sequences': all_sequences,
        'summary': {
            'total_unique_sequences': len(all_seqs)
        }
    }

def create_circuit_from_sequence(seq_ops: List[Dict[str, Any]], name: str = "circuit") -> QuantumCircuit:
    """Create a quantum circuit from operation sequence"""
    # Find all qubits used
    all_qubits = set()
    for op in seq_ops:
        all_qubits.update(op['qubits'])
    
    qubit_list = sorted(list(all_qubits))
    n_qubits = len(qubit_list)
    
    # Create mapping from physical to circuit qubits
    qubit_map = {phys: idx for idx, phys in enumerate(qubit_list)}
    
    # Create circuit
    qc = QuantumCircuit(n_qubits, name=name)
    
    # Add operations
    for op in seq_ops:
        op_name = op['operation']
        qubits = [qubit_map[q] for q in op['qubits']]
        params = op['params']
        
        if op_name == 'rz':
            qc.rz(params[0], qubits[0])
        elif op_name == 'sx':
            qc.sx(qubits[0])
        elif op_name == 'cz':
            qc.cz(qubits[0], qubits[1])
        elif op_name == 'x':
            qc.x(qubits[0])
        elif op_name == 'h':
            qc.h(qubits[0])
        else:
            print(f"  ⚠️  Unknown gate: {op_name}")
    
    # Add metadata
    qc.metadata = {
        'physical_qubits': qubit_list,
        'qubit_mapping': qubit_map,
        'operations': [{
            'operation': op['operation'],
            'qubits': list(op['qubits']),
            'params': list(op['params'])
        } for op in seq_ops]
    }
    
    return qc

def print_analysis(analysis: dict):
    """Print analysis results"""
    print("=" * 80)
    print("Sequence Analysis Summary")
    print("=" * 80)
    
    summary = analysis['summary']
    print(f"\n📊 Reports analyzed: {analysis['num_reports']}")
    print(f"📊 Total unique sequences: {summary['total_unique_sequences']}")
    
    # Top 10 sequences
    if analysis['all_sequences']:
        print(f"\n{'='*80}")
        print("📋 Top 10 Sequences (sorted by frequency)")
        print('='*80)
        for seq, data in list(analysis['all_sequences'].items())[:10]:
            print(f"  {seq}")
            print(f"    Count: {data['count']}, Reports: {data['reports']}")

def analyze_and_generate_circuits(data_dir: str = "delta_debug_data", output_dir: str = "tmp"):
    """
    Analyze problematic segments from delta debug reports and generate circuits.
    
    Args:
        data_dir: Directory containing delta_debug_report_*.json files (searches up to 2 levels deep)
        output_dir: Directory to save analysis results and generated circuits
    
    Returns:
        dict: Analysis results including sequences and statistics
    """
    circuits_dir = os.path.join(output_dir, "circuits")
    
    # Find report files (search up to 2 levels deep)
    if not os.path.exists(data_dir):
        print(f"❌ Directory '{data_dir}/' not found")
        return None
    
    # Search in current directory and one level deep
    report_files = []
    report_files.extend(glob.glob(os.path.join(data_dir, "delta_debug_report_*.json")))
    report_files.extend(glob.glob(os.path.join(data_dir, "*", "delta_debug_report_*.json")))
    report_files = sorted(set(report_files))  # Remove duplicates and sort
    
    if not report_files:
        print(f"❌ No delta_debug_report_*.json files found in '{data_dir}/' (searched 2 levels deep)")
        return None
    
    # Load reports
    print(f"Loading {len(report_files)} report(s) from {data_dir}/ (searched 2 levels deep)...")
    reports = []
    for fpath in report_files:
        try:
            reports.append(load_report(fpath))
            print(f"  ✓ Loaded: {fpath}")
        except Exception as e:
            print(f"  ⚠️  Skipped {fpath}: {e}")
    
    if not reports:
        print("❌ No reports loaded successfully")
        return None
    
    print(f"✓ Loaded {len(reports)} reports\n")
    
    # Analyze
    analysis = analyze_sequences(reports)
    print_analysis(analysis)
    
    # Save analysis to tmp directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean and recreate circuits directory to remove old files from previous runs
    if os.path.exists(circuits_dir):
        shutil.rmtree(circuits_dir)
    os.makedirs(circuits_dir)
    
    output_path = os.path.join(output_dir, "sequence_analysis.json")
    
    # Prepare JSON output for top 10 sequences (without operations objects)
    top_10_sequences = list(analysis['all_sequences'].items())[:10]
    json_output = {
        'summary': analysis['summary'],
        'top_10_sequences': {k: {'count': v['count'], 'reports': v['reports']} 
                            for k, v in top_10_sequences}
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"\n✅ Analysis saved to: {output_path}")
    
    # Generate circuits for top 10 sequences
    print(f"\n{'='*80}")
    print("🔧 Generating Quantum Circuits (Top 10)")
    print('='*80)
    
    circuits_generated = 0
    top_10_sequences = list(analysis['all_sequences'].items())[:10]
    for idx, (seq_name, data) in enumerate(top_10_sequences, 1):
        try:
            seq_ops = data['operations']
            circuit_name = f"seq_{idx:03d}"
            qc = create_circuit_from_sequence(seq_ops, name=circuit_name)
            
            # Save circuit as QPY (Qiskit's binary format)
            qpy_path = os.path.join(circuits_dir, f"{circuit_name}.qpy")
            with open(qpy_path, 'wb') as f:
                qpy.dump(qc, f)
            
            # Save circuit as QASM string
            qasm_path = os.path.join(circuits_dir, f"{circuit_name}.qasm")
            try:
                # Try OpenQASM 2.0
                from qiskit import qasm2
                qasm_str = qasm2.dumps(qc)
                with open(qasm_path, 'w') as f:
                    f.write(qasm_str)
            except Exception:
                # Fallback to simple string representation
                with open(qasm_path, 'w') as f:
                    f.write(str(qc))
            
            # Save circuit diagram as text
            diagram_path = os.path.join(circuits_dir, f"{circuit_name}_diagram.txt")
            with open(diagram_path, 'w') as f:
                f.write(f"Circuit: {seq_name}\n")
                f.write(f"Count: {data['count']}, Reports: {data['reports']}\n")
                f.write("="*60 + "\n")
                f.write(str(qc.draw(output='text', fold=-1)))
                f.write("\n\n")
                f.write("Physical Qubits: " + str(qc.metadata['physical_qubits']) + "\n")
            
            # Save circuit metadata
            metadata_path = os.path.join(circuits_dir, f"{circuit_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'sequence': seq_name,
                    'count': data['count'],
                    'reports': data['reports'],
                    'physical_qubits': qc.metadata['physical_qubits'],
                    'qubit_mapping': qc.metadata['qubit_mapping'],
                    'operations': qc.metadata['operations']
                }, f, indent=2)
            
            print(f"  ✓ {circuit_name}: {seq_name}")
            circuits_generated += 1
            
        except Exception as e:
            print(f"  ✗ Failed to generate circuit for: {seq_name}")
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Generated {circuits_generated} circuits in: {circuits_dir}/")
    print("   - QPY files: *.qpy (binary, can be loaded with qpy.load())")
    print("   - QASM files: *.qasm")
    print("   - Diagrams: *_diagram.txt")
    print("   - Metadata: *_metadata.json")
    
    return analysis
    
