# analyze_phantom_data.py
"""
Analyzes all phantom data and creates comparison plots
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
    for f in csv_files:
        df = pd.read_csv(f)
        if frequencies is None:
            frequencies = df['Frequency_Hz'].values / 1e9  # Converts to GHz
        data.append(df['S21_dB'].values)
    
    if data:
        return np.array(data), frequencies
    return None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_phantom_data.py <data_folder>")
        return
    
    base_folder = sys.argv[1]
    print(f"\n=== ANALYZING DATA FROM: {base_folder} ===\n")
    
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
        folder = f"{base_folder}/{condition}"
        if os.path.exists(folder):
            data, freqs = load_condition_data(folder)
            if data is not None:
                all_data[condition] = data
                if frequencies is None:
                    frequencies = freqs
                print(f"✅ Loaded {condition}: {data.shape[0]} paths, {data.shape[1]} points each")
            else:
                print(f"⚠️  No data found for {condition}")
        else:
            print(f"⚠️  Folder not found: {condition}")
    
    if not all_data:
        print("❌ No data loaded!")
        return
    
    # Create comparison plots
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
    plt.title('PULMO AI: Phantom Data Comparison\n(Average of 4 paths ± std dev)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim(-50, 0)
    plt.tight_layout()
    plt.savefig(f'{base_folder}/comparison_all_conditions.png', dpi=300)
    print(f"\n📊 Saved: comparison_all_conditions.png")
    
    # Plot 2: Individual paths for tumor condition
    if '04_tumor_phantom' in all_data:
        plt.figure(figsize=(14, 8))
        path_labels = ['Path 1 (1→3)', 'Path 2 (1→4)', 'Path 3 (2→3)', 'Path 4 (2→4)']
        path_colors = ['blue', 'green', 'orange', 'red']
        
        tumor_data = all_data['04_tumor_phantom']
        for i in range(min(4, tumor_data.shape[0])):
            plt.plot(frequencies, tumor_data[i], color=path_colors[i],
                    label=path_labels[i], linewidth=2)
        
        plt.xlabel('Frequency (GHz)', fontsize=12)
        plt.ylabel('S21 (dB)', fontsize=12)
        plt.title('PULMO AI: Tumor Phantom - All 4 Paths', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-50, 0)
        plt.tight_layout()
        plt.savefig(f'{base_folder}/tumor_all_paths.png', dpi=300)
        print("📊 Saved: tumor_all_paths.png")
    
    # Print statistics
    print("\n📊 KEY STATISTICS:")
    print("-" * 50)
    
    if '01_baseline_air' in all_data and '04_tumor_phantom' in all_data:
        baseline_avg = np.mean(all_data['01_baseline_air'])
        tumor_avg = np.mean(all_data['04_tumor_phantom'])
        total_drop = baseline_avg - tumor_avg
        
        print(f"Average S21 (all frequencies, all paths):")
        print(f"  Air Baseline:  {baseline_avg:.2f} dB")
        print(f"  Tumor Phantom: {tumor_avg:.2f} dB")
        print(f"  Total Drop:    {total_drop:.2f} dB")
        
        if total_drop >= 4.9:
            print("\n✅ TUMOR DETECTED! (>4.9 dB threshold)")
        else:
            print("\n⚠️  Tumor signal below threshold")
    
    # Calculate path-to-path variation for tumor
    if '04_tumor_phantom' in all_data:
        tumor_std = np.std(np.mean(all_data['04_tumor_phantom'], axis=1))
        print(f"\nPath-to-path variation (tumor): ±{tumor_std:.2f} dB")
        print("  (Higher variation = better spatial sensitivity)")
    
    print("\n✅ Analysis complete!")
    print(f"📁 All plots saved in: {base_folder}/")

if __name__ == "__main__":
    main()
