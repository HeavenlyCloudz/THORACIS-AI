# analyze_phantom_data.py
"""
Analyzes all phantom data and creates comparison plots
Run from the folder containing your phantom_data_* folders

Expected structure:
phantom_data_YYYYMMDD_HHMMSS/
├── 01_baseline_air/
│   ├── path1_rot0_*.csv
│   ├── path2_rot0_*.csv
│   ├── path3_rot0_*.csv
│   ├── path4_rot0_*.csv
│   ├── path1_rot120_*.csv
│   ├── path2_rot120_*.csv
│   ├── path3_rot120_*.csv
│   ├── path4_rot120_*.csv
│   ├── path1_rot240_*.csv
│   ├── path2_rot240_*.csv
│   ├── path3_rot240_*.csv
│   └── path4_rot240_*.csv
├── 02_healthy_phantom/
│   └── ... (12 files)
└── 03_tumor_phantom/
    └── ... (12 files)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from collections import defaultdict

def load_condition_data(folder, verbose=False):
    """
    Load all CSV files from a condition folder, grouped by path and rotation
    Returns: dict with keys (path_num, rotation) -> list of S21 arrays
    """
    data_by_path_rotation = defaultdict(list)
    frequencies = None
    
    csv_files = sorted(Path(folder).glob('*.csv'))
    if verbose:
        print(f"    Found {len(csv_files)} CSV files")
    
    for f in csv_files:
        df = pd.read_csv(f)
        if frequencies is None:
            frequencies = df['Frequency_Hz'].values / 1e9  # Convert to GHz
        
        s21 = df['S21_dB'].values
        
        # Extract path number and rotation from filename
        # Format: path1_rot0_20260322_143001.csv
        filename = f.stem
        parts = filename.split('_')
        
        # Find path number
        path_num = None
        for part in parts:
            if part.startswith('path'):
                path_num = int(part[4:])
                break
        
        # Find rotation
        rotation = None
        for part in parts:
            if part.startswith('rot'):
                rotation = int(part[3:])
                break
        
        if path_num and rotation is not None:
            data_by_path_rotation[(path_num, rotation)].append(s21)
        else:
            if verbose:
                print(f"    Warning: Could not parse {filename}")
    
    # Average multiple runs (if any) for each (path, rotation) combination
    averaged_data = {}
    for key, s21_list in data_by_path_rotation.items():
        averaged_data[key] = np.mean(s21_list, axis=0) if len(s21_list) > 1 else s21_list[0]
    
    if verbose:
        print(f"    Found {len(averaged_data)} unique (path, rotation) combinations")
    
    return averaged_data, frequencies

def group_by_path(data_dict, num_paths=4):
    """
    Group data by path number across all rotations
    Returns: dict path_num -> list of S21 arrays (one per rotation)
    """
    path_data = {p: [] for p in range(1, num_paths + 1)}
    for (path_num, rotation), s21 in data_dict.items():
        path_data[path_num].append(s21)
    return path_data

def group_by_rotation(data_dict, rotations=[0, 120, 240]):
    """
    Group data by rotation across all paths
    Returns: dict rotation -> list of S21 arrays (one per path)
    """
    rotation_data = {r: [] for r in rotations}
    for (path_num, rotation), s21 in data_dict.items():
        if rotation in rotation_data:
            rotation_data[rotation].append(s21)
    return rotation_data

def create_path_labels(num_paths=4):
    """Create descriptive labels for paths"""
    path_labels = {
        1: 'Path 1 (1→3)',
        2: 'Path 2 (1→4)',
        3: 'Path 3 (2→3)',
        4: 'Path 4 (2→4)'
    }
    return [path_labels[i] for i in range(1, num_paths + 1)]

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
    print(f" PULMO AI - PHANTOM DATA ANALYSIS")
    print(f" {base_folder}")
    print(f"{'='*70}\n")
    
    # Define conditions (UPDATED: 3 conditions only)
    conditions = [
        ('01_baseline_air', 'Air Baseline', '#1f77b4'),      # Blue
        ('02_healthy_phantom', 'Healthy Phantom', '#2ca02c'),  # Green
        ('03_tumor_phantom', 'Tumor Phantom', '#d62728')       # Red
    ]
    
    # Load all data
    all_data = {}
    frequencies = None
    rotations_found = set()
    
    print("📂 Loading data:")
    for folder_name, display_name, color in conditions:
        folder = os.path.join(base_folder, folder_name)
        if os.path.exists(folder):
            print(f"   {display_name}...")
            data, freqs = load_condition_data(folder, verbose=True)
            if data:
                all_data[folder_name] = {
                    'display_name': display_name,
                    'color': color,
                    'data': data,
                    'freqs': freqs
                }
                if frequencies is None:
                    frequencies = freqs
                # Collect rotations found
                for (_, rotation) in data.keys():
                    rotations_found.add(rotation)
            else:
                print(f"   ⚠️  No data found for {display_name}")
        else:
            print(f"   ⚠️  Folder not found: {folder_name}")
    
    if not all_data:
        print("❌ No data loaded!")
        return
    
    rotations = sorted(rotations_found)
    print(f"\n📐 Rotations detected: {rotations}")
    
    # Create output folder for plots
    plots_folder = os.path.join(base_folder, 'analysis_plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    print("\n📊 Generating plots...")
    
    # =========================================================
    # PLOT 1: All conditions averaged across all paths and rotations
    # =========================================================
    plt.figure(figsize=(14, 8))
    
    for folder_name, info in all_data.items():
        data_dict = info['data']
        # Average all scans in this condition
        all_scans = list(data_dict.values())
        avg_data = np.mean(all_scans, axis=0)
        std_data = np.std(all_scans, axis=0)
        
        plt.plot(frequencies, avg_data, color=info['color'],
                label=info['display_name'], linewidth=2.5)
        plt.fill_between(frequencies, avg_data - std_data, avg_data + std_data,
                        color=info['color'], alpha=0.2)
    
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('S21 (dB)', fontsize=12)
    plt.title('PULMO AI: Phantom Comparison (All Paths & Rotations Averaged)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.ylim(-50, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, '01_all_conditions_averaged.png'), dpi=300)
    plt.close()
    print(f"   ✅ 01_all_conditions_averaged.png")
    
    # =========================================================
    # PLOT 2: Tumor phantom - individual paths averaged across rotations
    # =========================================================
    if '03_tumor_phantom' in all_data:
        plt.figure(figsize=(14, 8))
        tumor_data = all_data['03_tumor_phantom']['data']
        path_data = group_by_path(tumor_data, num_paths=4)
        path_labels = create_path_labels()
        path_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for path_num in range(1, 5):
            if path_data[path_num]:
                avg_path = np.mean(path_data[path_num], axis=0)
                std_path = np.std(path_data[path_num], axis=0)
                plt.plot(frequencies, avg_path, color=path_colors[path_num-1],
                        label=path_labels[path_num-1], linewidth=2)
                plt.fill_between(frequencies, avg_path - std_path, avg_path + std_path,
                                color=path_colors[path_num-1], alpha=0.2)
        
        plt.xlabel('Frequency (GHz)', fontsize=12)
        plt.ylabel('S21 (dB)', fontsize=12)
        plt.title('PULMO AI: Tumor Phantom - Individual Paths (Averaged Across Rotations)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.ylim(-50, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '02_tumor_by_path.png'), dpi=300)
        plt.close()
        print(f"   ✅ 02_tumor_by_path.png")
    
    # =========================================================
    # PLOT 3: Tumor phantom - rotations (shows spatial sensitivity)
    # =========================================================
    if '03_tumor_phantom' in all_data:
        plt.figure(figsize=(14, 8))
        tumor_data = all_data['03_tumor_phantom']['data']
        rotation_data = group_by_rotation(tumor_data, rotations=rotations)
        
        rotation_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for rot_idx, rotation in enumerate(rotations):
            if rotation_data[rotation]:
                avg_rot = np.mean(rotation_data[rotation], axis=0)
                std_rot = np.std(rotation_data[rotation], axis=0)
                plt.plot(frequencies, avg_rot, color=rotation_colors[rot_idx % len(rotation_colors)],
                        label=f'Rotation {rotation}°', linewidth=2)
                plt.fill_between(frequencies, avg_rot - std_rot, avg_rot + std_rot,
                                color=rotation_colors[rot_idx % len(rotation_colors)], alpha=0.2)
        
        plt.xlabel('Frequency (GHz)', fontsize=12)
        plt.ylabel('S21 (dB)', fontsize=12)
        plt.title('PULMO AI: Tumor Phantom - Effect of Rotation (Spatial Sensitivity)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.ylim(-50, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '03_tumor_by_rotation.png'), dpi=300)
        plt.close()
        print(f"   ✅ 03_tumor_by_rotation.png")
    
    # =========================================================
    # PLOT 4: Baseline vs Tumor - Best detection frequency
    # =========================================================
    if '01_baseline_air' in all_data and '03_tumor_phantom' in all_data:
        plt.figure(figsize=(14, 8))
        
        baseline_data = all_data['01_baseline_air']['data']
        tumor_data = all_data['03_tumor_phantom']['data']
        
        # Average all scans for each condition
        baseline_avg = np.mean(list(baseline_data.values()), axis=0)
        tumor_avg = np.mean(list(tumor_data.values()), axis=0)
        baseline_std = np.std(list(baseline_data.values()), axis=0)
        tumor_std = np.std(list(tumor_data.values()), axis=0)
        
        plt.plot(frequencies, baseline_avg, color='#1f77b4', 
                label='Air Baseline', linewidth=2.5)
        plt.fill_between(frequencies, baseline_avg - baseline_std, baseline_avg + baseline_std,
                        color='#1f77b4', alpha=0.2)
        
        plt.plot(frequencies, tumor_avg, color='#d62728', 
                label='Tumor Phantom', linewidth=2.5)
        plt.fill_between(frequencies, tumor_avg - tumor_std, tumor_avg + tumor_std,
                        color='#d62728', alpha=0.2)
        
        # Calculate difference
        diff = baseline_avg - tumor_avg
        best_idx = np.argmax(diff)
        best_freq = frequencies[best_idx]
        best_drop = diff[best_idx]
        
        plt.axvline(x=best_freq, color='purple', linestyle='--', alpha=0.7,
                   label=f'Best Detection: {best_freq:.2f} GHz ({best_drop:.2f} dB drop)')
        
        # Add threshold line
        threshold = 4.9  # Your proven threshold
        plt.axhline(y=baseline_avg[best_idx] - threshold, 
                   color='red', linestyle=':', alpha=0.7, linewidth=2,
                   label=f'Tumor Threshold ({threshold} dB)')
        
        plt.xlabel('Frequency (GHz)', fontsize=12)
        plt.ylabel('S21 (dB)', fontsize=12)
        plt.title('PULMO AI: Baseline vs Tumor Detection', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.ylim(-50, 0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '04_baseline_vs_tumor.png'), dpi=300)
        plt.close()
        print(f"   ✅ 04_baseline_vs_tumor.png")
    
    # =========================================================
    # STATISTICS
    # =========================================================
    print("\n" + "="*70)
    print("📊 STATISTICS SUMMARY")
    print("="*70)
    
    # Per condition statistics
    print("\n📈 Per Condition Averages:")
    print("-" * 50)
    for folder_name, info in all_data.items():
        data_dict = info['data']
        all_values = np.concatenate(list(data_dict.values()))
        print(f"\n{info['display_name']}:")
        print(f"   Mean S21: {np.mean(all_values):.2f} ± {np.std(all_values):.2f} dB")
        print(f"   Range: {np.min(all_values):.2f} to {np.max(all_values):.2f} dB")
        print(f"   Total scans: {len(data_dict)} (paths × rotations)")
    
    # Tumor detection analysis
    if '01_baseline_air' in all_data and '03_tumor_phantom' in all_data:
        baseline_values = np.concatenate(list(all_data['01_baseline_air']['data'].values()))
        tumor_values = np.concatenate(list(all_data['03_tumor_phantom']['data'].values()))
        
        baseline_mean = np.mean(baseline_values)
        tumor_mean = np.mean(tumor_values)
        total_drop = baseline_mean - tumor_mean
        
        print(f"\n{'='*70}")
        print("🎯 TUMOR DETECTION ANALYSIS")
        print("="*70)
        print(f"Air Baseline Average:  {baseline_mean:.2f} dB")
        print(f"Tumor Phantom Average: {tumor_mean:.2f} dB")
        print(f"Total Signal Drop:     {total_drop:.2f} dB")
        
        if total_drop >= 4.9:
            print("\n✅✅✅ TUMOR DETECTED! (>4.9 dB threshold) ✅✅✅")
        else:
            print(f"\n⚠️  Tumor signal ({total_drop:.2f} dB) below 4.9 dB threshold")
            print("   Check: Is tumor simulant properly embedded?")
        
        # Best frequency detection
        baseline_avg_freq = np.mean(list(all_data['01_baseline_air']['data'].values()), axis=0)
        tumor_avg_freq = np.mean(list(all_data['03_tumor_phantom']['data'].values()), axis=0)
        diff = baseline_avg_freq - tumor_avg_freq
        best_idx = np.argmax(diff)
        
        print(f"\n📡 Best Detection Frequency: {frequencies[best_idx]:.3f} GHz")
        print(f"   Signal drop at this frequency: {diff[best_idx]:.2f} dB")
    
    # Path-to-path variation for tumor
    if '03_tumor_phantom' in all_data:
        tumor_data = all_data['03_tumor_phantom']['data']
        path_data = group_by_path(tumor_data, num_paths=4)
        
        path_means = []
        for path_num in range(1, 5):
            if path_data[path_num]:
                path_means.append(np.mean(path_data[path_num]))
        
        if path_means:
            path_std = np.std(path_means)
            print(f"\n📐 Path-to-path variation: ±{path_std:.2f} dB")
            if path_std > 2:
                print("   ✅ Good spatial sensitivity - tumor affects different paths differently")
            else:
                print("   ⚠️  Low spatial variation - tumor may be centered or small")
    
    # Rotation variation for tumor
    if '03_tumor_phantom' in all_data and len(rotations) > 1:
        tumor_data = all_data['03_tumor_phantom']['data']
        rotation_data = group_by_rotation(tumor_data, rotations=rotations)
        
        rotation_means = []
        for rotation in rotations:
            if rotation_data[rotation]:
                rotation_means.append(np.mean(rotation_data[rotation]))
        
        if rotation_means:
            rotation_std = np.std(rotation_means)
            print(f"\n🔄 Rotation-to-rotation variation: ±{rotation_std:.2f} dB")
            if rotation_std > 2:
                print("   ✅ Good spatial sensitivity - tumor position affects signal")
            else:
                print("   ⚠️  Low rotation variation - tumor may be centered")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 All plots saved in: {plots_folder}/")
    print("="*70)

if __name__ == "__main__":
    main()
