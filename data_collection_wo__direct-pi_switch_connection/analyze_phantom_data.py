# analyze_phantom_data.py
"""
Analyzes all phantom data and creates comparison plots
Run from the folder containing your phantom_data_* folders
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

def load_condition_data(folder):
    """Load all CSV files from a condition folder"""
    data = []
    frequencies = None
    
    csv_files = sorted(Path(folder).glob('*.csv'))
    print(f"    Found {len(csv_files)} CSV files")
    
    for f in csv_files:
        df = pd.read_csv(f)
        if frequencies is None:
            frequencies = df['Frequency_Hz'].values / 1e9  # Convert to GHz
        data.append(df['S21_dB'].values)
    
    if data:
        return np.array(data), frequencies
    return None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_phantom_data.py <data_folder>")
        print("Example: python analyze_phantom_data.py phantom_data_20260322_172408")
        return
    
    base_folder = sys.argv[1]
    
    # Check if folder exists
    if not os.path.exists(base_folder):
        print(f"❌ Folder not found: {base_folder}")
        print(f"   Make sure you're in the right directory")
        return
    
    print(f"\n{'='*70}")
    print(f" ANALYZING DATA FROM: {base_folder}")
    print(f"{'='*70}\n")
    
    # Define conditions in order
    conditions = [
        '01_baseline_air',
        '02_healthy_phantom_1',
        '03_healthy_phantom_2',
        '04_tumor_phantom'
    ]
    
    # Load all data
    all_data = {}
    frequencies = None
    
    for condition in conditions:
        folder = os.path.join(base_folder, condition)
        if os.path.exists(folder):
            print(f"📂 Loading {condition}...")
            data, freqs = load_condition_data(folder)
            if data is not None:
                all_data[condition] = data
                if frequencies is None:
                    frequencies = freqs
                print(f"   ✅ {data.shape[0]} scans, {data.shape[1]} points each")
                print(f"   📊 S21 range: {np.min(data):.1f} to {np.max(data):.1f} dB")
            else:
                print(f"   ⚠️  No data found")
        else:
            print(f"⚠️  Folder not found: {condition}")
    
    if not all_data:
        print("❌ No data loaded!")
        return
    
    # Create comparison plots
    print("\n📊 Generating plots...")
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['Air Baseline', 'Healthy #1', 'Healthy #2', 'Tumor']
    
    # Plot 1: All conditions averaged across paths
    plt.figure(figsize=(14, 8))
    
    for idx, (condition, data) in enumerate(all_data.items()):
        avg_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        
        plt.plot(frequencies, avg_data, color=colors[idx], 
                label=labels[idx], linewidth=2)
        plt.fill_between(frequencies, avg_data - std_data, avg_data + std_data,
                        color=colors[idx], alpha=0.2)
    
    # Mark tumor threshold
    if '01_baseline_air' in all_data and '04_tumor_phantom' in all_data:
        baseline_avg = np.mean(all_data['01_baseline_air'], axis=0)
        tumor_avg = np.mean(all_data['04_tumor_phantom'], axis=0)
        
        # Find best detection frequency
        diff = baseline_avg - tumor_avg
        best_idx = np.argmax(diff)
        best_freq = frequencies[best_idx]
        best_drop = diff[best_idx]
        
        plt.axvline(x=best_freq, color='purple', linestyle='--', alpha=0.5,
                   label=f'Best Detection: {best_freq:.2f} GHz ({best_drop:.2f} dB drop)')
        
        # Add threshold line
        threshold = 4.9  # Your proven threshold
        plt.axhline(y=baseline_avg[best_idx] - threshold, 
                   color='red', linestyle=':', alpha=0.5,
                   label=f'Tumor Threshold ({threshold} dB)')
    
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('S21 (dB)', fontsize=12)
    plt.title('PULMO AI: Phantom Data Comparison\n(Average of all scans ± std dev)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim(-50, 0)
    plt.tight_layout()
    plot1_path = os.path.join(base_folder, 'comparison_all_conditions.png')
    plt.savefig(plot1_path, dpi=300)
    print(f"✅ Saved: {plot1_path}")
    plt.close()
    
    # Plot 2: Individual paths for tumor condition
    if '04_tumor_phantom' in all_data:
        plt.figure(figsize=(14, 8))
        path_labels = ['Path 1 (1→3)', 'Path 2 (1→4)', 'Path 3 (2→3)', 'Path 4 (2→4)']
        path_colors = ['blue', 'green', 'orange', 'red']
        
        tumor_data = all_data['04_tumor_phantom']
        # Group by path (assuming files are in order: path1, path2, path3, path4, then rotations)
        # For each rotation, we have 4 files. Let's average across rotations for each path
        num_rotations = tumor_data.shape[0] // 4
        print(f"\n   Found {num_rotations} rotations (3 expected)")
        
        for path_idx in range(4):
            # Average all scans for this path across rotations
            path_scans = tumor_data[path_idx::4]  # Every 4th file starting at path_idx
            avg_path = np.mean(path_scans, axis=0)
            plt.plot(frequencies, avg_path, color=path_colors[path_idx],
                    label=path_labels[path_idx], linewidth=2)
        
        plt.xlabel('Frequency (GHz)', fontsize=12)
        plt.ylabel('S21 (dB)', fontsize=12)
        plt.title('PULMO AI: Tumor Phantom - All 4 Paths (Averaged Across Rotations)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-50, 0)
        plt.tight_layout()
        plot2_path = os.path.join(base_folder, 'tumor_all_paths.png')
        plt.savefig(plot2_path, dpi=300)
        print(f"✅ Saved: {plot2_path}")
        plt.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("📊 KEY STATISTICS")
    print("="*70)
    
    # Calculate averages for each condition
    for condition in ['01_baseline_air', '02_healthy_phantom_1', '03_healthy_phantom_2', '04_tumor_phantom']:
        if condition in all_data:
            avg_all = np.mean(all_data[condition])
            std_all = np.std(all_data[condition])
            name = condition.replace('_', ' ').title()
            print(f"\n{name}:")
            print(f"  Mean S21: {avg_all:.2f} ± {std_all:.2f} dB")
            print(f"  Min: {np.min(all_data[condition]):.2f} dB")
            print(f"  Max: {np.max(all_data[condition]):.2f} dB")
    
    # Tumor detection analysis
    if '01_baseline_air' in all_data and '04_tumor_phantom' in all_data:
        baseline_all = np.mean(all_data['01_baseline_air'])
        tumor_all = np.mean(all_data['04_tumor_phantom'])
        total_drop = baseline_all - tumor_all
        
        print(f"\n{'='*70}")
        print("🎯 TUMOR DETECTION ANALYSIS")
        print("="*70)
        print(f"Air Baseline Average:  {baseline_all:.2f} dB")
        print(f"Tumor Phantom Average: {tumor_all:.2f} dB")
        print(f"Total Signal Drop:     {total_drop:.2f} dB")
        
        if total_drop >= 4.9:
            print("\n✅✅✅ TUMOR DETECTED! (>4.9 dB threshold) ✅✅✅")
        else:
            print(f"\n⚠️  Tumor signal ({total_drop:.2f} dB) below 4.9 dB threshold")
            print("   Check: Is tumor simulant properly embedded?")
    
    # Path-to-path variation
    if '04_tumor_phantom' in all_data:
        path_means = []
        for path_idx in range(4):
            path_scans = all_data['04_tumor_phantom'][path_idx::4]
            path_means.append(np.mean(path_scans))
        path_std = np.std(path_means)
        print(f"\nPath-to-path variation: ±{path_std:.2f} dB")
        if path_std > 2:
            print("  ✅ Good spatial sensitivity - tumor affects different paths differently")
        else:
            print("  ⚠️  Low spatial variation - tumor may be centered or small")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 All plots saved in: {base_folder}/")
    print(f"   - comparison_all_conditions.png")
    print(f"   - tumor_all_paths.png")
    print("="*70)

if __name__ == "__main__":
    main()
