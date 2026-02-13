#!/usr/bin/env python3
"""
Visualize circuit verification results with operation sequences
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_results(tmp_dir='tmp'):
    """Load comparison results and sequence analysis"""
    try:
        with open(Path(tmp_dir) / 'circuit_comparison_results.json', 'r') as f:
            comparison_results = json.load(f)
        
        with open(Path(tmp_dir) / 'sequence_analysis.json', 'r') as f:
            sequence_analysis = json.load(f)
        
        return comparison_results, sequence_analysis
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return None, None


def get_sequence_for_circuit(circuit_name, sequence_analysis):
    """
    Get the operation sequence for a given circuit
    circuit_name format: seq_001.qpy -> look for "seq_001" in sequences
    """
    # Extract circuit index (e.g., "seq_001" from "seq_001.qpy")
    circuit_base = circuit_name.replace('.qpy', '')
    
    # Find matching sequence
    sequences = sequence_analysis.get('top_10_sequences', {})
    
    # Get the index (1-indexed in the dict keys)
    for idx, (seq_name, seq_data) in enumerate(sequences.items(), 1):
        if circuit_base == f"seq_{idx:03d}":
            return seq_name, seq_data
    
    return None, None


def shorten_sequence(seq_name, max_length=30):
    """Shorten sequence name for display"""
    if len(seq_name) <= max_length:
        return seq_name
    
    # Try to show first and last operations
    parts = seq_name.split(' → ')
    if len(parts) <= 2:
        return seq_name[:max_length-3] + '...'
    
    # Show first and last operation
    return f"{parts[0]} → ... → {parts[-1]}"


def visualize_tvd_with_sequences(comparison_results, sequence_analysis, output_dir='tmp', show_plot=False):
    """
    Create visualization with circuit names and operation sequences
    
    Args:
        comparison_results: Circuit comparison results
        sequence_analysis: Sequence analysis data
        output_dir: Directory to save the figure
        show_plot: Whether to display the plot (default: False, saves only)
    """
    # Prepare data
    data = []
    for r in sorted(comparison_results, key=lambda x: x['file']):
        circuit_name = r['file']
        seq_name, seq_data = get_sequence_for_circuit(circuit_name, sequence_analysis)
        
        # Shorten sequence for display
        if seq_name:
            seq_display = shorten_sequence(seq_name, max_length=40)
        else:
            seq_display = "N/A"
        
        data.append({
            'circuit': circuit_name,
            'sequence': seq_display,
            'sequence_full': seq_name if seq_name else "N/A",
            'tvd_loss': r['tvd_loss'],
            'count': seq_data.get('count', 0) if seq_data else 0,
            'reports': seq_data.get('reports', []) if seq_data else []
        })
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Bar plot
    bars = ax.bar(range(len(df)), df['tvd_loss'], color='steelblue', alpha=0.8, edgecolor='black')
    
    # Color bars by TVD loss (gradient)
    max_tvd = df['tvd_loss'].max()
    for i, (bar, tvd) in enumerate(zip(bars, df['tvd_loss'])):
        # Color gradient from green (low TVD) to red (high TVD)
        ratio = tvd / max_tvd if max_tvd > 0 else 0
        bar.set_color(plt.cm.RdYlGn_r(ratio))
    
    # Set labels
    ax.set_ylabel('TVD Loss', fontsize=12, fontweight='bold')
    ax.set_title('Circuit Verification Results: TVD Loss by Operation Sequence', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # X-axis labels (two lines)
    ax.set_xticks(range(len(df)))
    
    # Create two-line labels
    labels = []
    for _, row in df.iterrows():
        # Line 1: Circuit name
        # Line 2: Operation sequence
        label = f"{row['circuit']}\n{row['sequence']}"
        labels.append(label)
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, tvd) in enumerate(zip(bars, df['tvd_loss'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{tvd:.4f}',
                ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'tvd_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close figure to free memory
    
    return df


def print_detailed_summary(df):
    """Print detailed text summary"""
    print("\n" + "="*100)
    print("📊 Detailed Circuit Verification Summary")
    print("="*100)
    
    for idx, row in df.iterrows():
        print(f"\n🔸 {row['circuit']}")
        print(f"   Operation Sequence: {row['sequence_full']}")
        print(f"   TVD Loss: {row['tvd_loss']:.6f}")
        if row['count'] > 0:
            print(f"   Frequency: {row['count']} times in reports {row['reports']}")
    
    print("\n" + "="*100)
    print("📈 Statistics")
    print("="*100)
    print(f"Total Circuits: {len(df)}")
    print(f"Average TVD Loss: {df['tvd_loss'].mean():.6f}")
    print(f"Max TVD Loss: {df['tvd_loss'].max():.6f} ({df.loc[df['tvd_loss'].idxmax(), 'circuit']})")
    print(f"Min TVD Loss: {df['tvd_loss'].min():.6f} ({df.loc[df['tvd_loss'].idxmin(), 'circuit']})")
    print(f"Std Dev: {df['tvd_loss'].std():.6f}")


def main():
    """Main function to visualize results"""
    print("🔍 Loading results...")
    comparison_results, sequence_analysis = load_results('tmp')
    
    if not comparison_results or not sequence_analysis:
        print("❌ Failed to load results. Please run the analysis and verification first.")
        return
    
    print(f"✓ Loaded {len(comparison_results)} circuit results\n")
    
    # Visualize
    df = visualize_tvd_with_sequences(comparison_results, sequence_analysis)
    
    # Print summary
    print_detailed_summary(df)
    
    # Save summary to CSV
    output_csv = Path('tmp') / 'tvd_summary.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Summary saved to: {output_csv}")


if __name__ == "__main__":
    main()
