# analyze_phantom_data_backsub.py - UPDATED with Background Subtraction
"""
Analyzes all phantom data with BACKGROUND SUBTRACTION to remove direct antenna coupling
Now only uses Path 1 (1→3) and Path 2 (2→4) - the aligned opposite pairs

Run: python analyze_phantom_data_backsub.py phantom_data_YYYYMMDD_HHMMSS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from collections import defaultdict

# CONFIGURATION - Only use these paths (aligned opposite pairs)
VALID_PATHS = [1, 2]  # Path 1: 1→3, Path 2: 2→4
PATH_LABELS = {
    1: 'Path 1 (1→3) - Opposite, Aligned',
    2: 'Path 2 (2→4) - Opposite, Aligned',
}
PATH_COLORS = {
    1: '#1f77b4',  # Blue
    2: '#ff7f0e',  # Orange
}

def db_to_linear(db):
    """Convert dB to linear magnitude (power)"""
    return 10 ** (db / 10)

def linear_to_db(linear):
    """Convert linear magnitude to dB, with floor to avoid -inf"""
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def load_condition_data_with_background(folder, baseline_data_dict=None, verbose=False):
    """
    Load all CSV files from a condition folder
    If baseline_data_dict is provided, apply background subtraction
    """
    data_by_path_rotation = defaultdict(list)
    frequencies = None
    
    csv_files = sorted(Path(folder).glob('*.csv'))
    if verbose:
        print(f"    Found {len(csv_files)} CSV files")
    
    for f in csv_files:
        df = pd.read_csv(f)
        if frequencies is None:
            frequencies = df['Frequency_Hz'].values / 1e9
        
        s21_db = df['S21_dB'].values
        
        # Extract path number and rotation from filename
        filename = f.stem
        parts = filename.split('_')
        
        path_num = None
        for part in parts:
            if part.startswith('path'):
                path_num = int(part[4:])
                break
        
        rotation = None
        for part in parts:
            if part.startswith('rot'):
                rotation = int(part[3:])
                break
        
        # ONLY use valid paths (1 and 2)
        if path_num and rotation is not None and path_num in VALID_PATHS:
            # Apply background subtraction if baseline provided
            if baseline_data_dict and (path_num, rotation) in baseline_data_dict:
                baseline_linear = db_to_linear(baseline_data_dict[(path_num, rotation)])
                signal_linear = db_to_linear(s21_db)
                corrected_linear = signal_linear - baseline_linear
                s21_db = linear_to_db(corrected_linear)
            
            data_by_path_rotation[(path_num, rotation)].append(s21_db)
    
    # Average multiple runs
    averaged_data = {}
    for key, s21_list in data_by_path_rotation.items():
        averaged_data[key] = np.mean(s21_list, axis=0) if len(s21_list) > 1 else s21_list[0]
    
    if verbose:
        print(f"    Loaded {len(averaged_data)} (path, rotation) combinations")
    
    return averaged_data, frequencies

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_phantom_data_backsub.py <data_folder>")
        print("Example: python analyze_phantom_data_backsub.py phantom_data_20260324_154658")
        return
    
    base_folder = sys.argv[1]
    
    if not os.path.exists(base_folder):
        print(f"❌ Folder not found: {base_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f" PULMO AI - PHANTOM DATA ANALYSIS (WITH BACKGROUND SUBTRACTION)")
    print(f" {base_folder}")
    print(f"{'='*70}\n")
    print(f"📡 USING ONLY PATHS {VALID_PATHS} (aligned opposite pairs)")
    print(f"   Path 1: Antenna 1 → Antenna 3")
    print(f"   Path 2: Antenna 2 → Antenna 4")
    
    conditions = [
        ('01_baseline_air', 'Air Baseline', '#1f77b4'),
        ('02_healthy_phantom', 'Healthy Phantom', '#2ca02c'),
        ('03_tumor_phantom', 'Tumor Phantom', '#d62728')
    ]
    
    # First, load baseline (air) data WITHOUT subtraction
    print("\n📂 Loading Air Baseline (for background subtraction)...")
    baseline_folder = os.path.join(base_folder, '01_baseline_air')
    baseline_data, frequencies = load_condition_data_with_background(
        baseline_folder, baseline_data_dict=None, verbose=True
    )
    
    if not baseline_data:
        print("❌ No baseline data found!")
        return
    
    print(f"\n📂 Loading conditions with background subtraction...")
    
    all_data = {}
    rotations_found = set()
    
    for folder_name, display_name, color in conditions:
        folder = os.path.join(base_folder, folder_name)
        if os.path.exists(folder):
            print(f"   {display_name}...")
            # For baseline, use raw data; for others, subtract baseline
            if folder_name == '01_baseline_air':
                data, freqs = load_condition_data_with_background(
                    folder, baseline_data_dict=None, verbose=True
                )
            else:
                data, freqs = load_condition_data_with_background(
                    folder, baseline_data_dict=baseline_data, verbose=True
                )
            
            if data:
                all_data[folder_name] = {
                    'display_name': display_name,
                    'color': color,
                    'data': data,
                    'freqs': freqs
                }
                if frequencies is None:
                    frequencies = freqs
                for (_, rotation) in data.keys():
                    rotations_found.add(rotation)
        else:
            print(f"   ⚠️  Folder not found: {folder_name}")
    
    if not all_data:
        print("❌ No data loaded!")
        return
    
    rotations = sorted(rotations_found)
    print(f"\n📐 Rotations detected: {rotations}")
    
    plots_folder = os.path.join(base_folder, 'analysis_plots_subtracted')
    os.makedirs(plots_folder, exist_ok=True)
    
    print("\n📊 Generating plots...")
    
    # =========================================================
    # PLOT 1: All conditions (after subtraction)
    # =========================================================
    plt.figure(figsize=(14, 8))
    
    for folder_name, info in all_data.items():
        data_dict = info['data']
        all_scans = list(data_dict.values())
        avg_data = np.mean(all_scans, axis=0)
        std_data = np.std(all_scans, axis=0)
        
        plt.plot(frequencies, avg_data, color=info['color'],
                label=info['display_name'], linewidth=2.5)
        plt.fill_between(frequencies, avg_data - std_data, avg_data + std_data,
                        color=info['color'], alpha=0.2)
    
    plt.xlabel('Frequency (GHz)', fontsize=12)
    plt.ylabel('S21 (dB)', fontsize=12)
    plt.title('PULMO AI: Phantom Comparison (After Background Subtraction)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, '01_all_conditions_subtracted.png'), dpi=300)
    plt.close()
    print(f"   ✅ 01_all_conditions_subtracted.png")
    
    # =========================================================
    # PLOT 2: Path-by-path comparison (Tumor vs Healthy)
    # =========================================================
    if '02_healthy_phantom' in all_data and '03_tumor_phantom' in all_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        healthy_data = all_data['02_healthy_phantom']['data']
        tumor_data = all_data['03_tumor_phantom']['data']
        
        for path_num in VALID_PATHS:
            # Get data for this path across rotations
            healthy_path = []
            tumor_path = []
            for rotation in rotations:
                key_h = (path_num, rotation)
                key_t = (path_num, rotation)
                if key_h in healthy_data:
                    healthy_path.append(healthy_data[key_h])
                if key_t in tumor_data:
                    tumor_path.append(tumor_data[key_t])
            
            if healthy_path and tumor_path:
                healthy_avg = np.mean(healthy_path, axis=0)
                tumor_avg = np.mean(tumor_path, axis=0)
                
                ax = axes[0]
                ax.plot(frequencies, healthy_avg, color=PATH_COLORS[path_num],
                       label=f'{PATH_LABELS[path_num]} - Healthy', linewidth=2, linestyle='--')
                ax.plot(frequencies, tumor_avg, color=PATH_COLORS[path_num],
                       label=f'{PATH_LABELS[path_num]} - Tumor', linewidth=2)
                
                # Difference plot
                diff = healthy_avg - tumor_avg
                ax = axes[1]
                ax.plot(frequencies, diff, color=PATH_COLORS[path_num],
                       label=PATH_LABELS[path_num], linewidth=2)
        
        axes[0].set_xlabel('Frequency (GHz)')
        axes[0].set_ylabel('S21 (dB)')
        axes[0].set_title('Healthy vs Tumor (After Subtraction)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Frequency (GHz)')
        axes[1].set_ylabel('Signal Drop (dB)')
        axes[1].set_title('Tumor Detection Signal (Healthy - Tumor)')
        axes[1].axhline(y=4.9, color='red', linestyle='--', label='Detection Threshold (4.9 dB)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, '02_path_comparison_subtracted.png'), dpi=300)
        plt.close()
        print(f"   ✅ 02_path_comparison_subtracted.png")
    
    # =========================================================
    # STATISTICS WITH DETECTION
    # =========================================================
    print("\n" + "="*70)
    print("📊 STATISTICS SUMMARY (AFTER BACKGROUND SUBTRACTION)")
    print("="*70)
    
    if '02_healthy_phantom' in all_data and '03_tumor_phantom' in all_data:
        healthy_data = all_data['02_healthy_phantom']['data']
        tumor_data = all_data['03_tumor_phantom']['data']
        
        all_healthy = np.concatenate(list(healthy_data.values()))
        all_tumor = np.concatenate(list(tumor_data.values()))
        
        print(f"\n📈 Overall Averages:")
        print(f"   Healthy Phantom: {np.mean(all_healthy):.2f} ± {np.std(all_healthy):.2f} dB")
        print(f"   Tumor Phantom:   {np.mean(all_tumor):.2f} ± {np.std(all_tumor):.2f} dB")
        print(f"   Difference:      {np.mean(all_healthy) - np.mean(all_tumor):.2f} dB")
        
        # Best detection by path
        print(f"\n🎯 BEST DETECTION BY PATH:")
        best_overall = 0
        best_overall_freq = 0
        best_overall_path = 0
        
        for path_num in VALID_PATHS:
            healthy_path = []
            tumor_path = []
            for rotation in rotations:
                if (path_num, rotation) in healthy_data:
                    healthy_path.append(healthy_data[(path_num, rotation)])
                if (path_num, rotation) in tumor_data:
                    tumor_path.append(tumor_data[(path_num, rotation)])
            
            if healthy_path and tumor_path:
                healthy_avg = np.mean(healthy_path, axis=0)
                tumor_avg = np.mean(tumor_path, axis=0)
                diff = healthy_avg - tumor_avg
                best_idx = np.argmax(diff)
                best_freq = frequencies[best_idx]
                best_drop = diff[best_idx]
                
                print(f"\n   {PATH_LABELS[path_num]}:")
                print(f"      Best frequency: {best_freq:.3f} GHz")
                print(f"      Signal drop: {best_drop:.2f} dB")
                
                if best_drop > best_overall:
                    best_overall = best_drop
                    best_overall_freq = best_freq
                    best_overall_path = path_num
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST OVERALL DETECTION:")
        print(f"   Path {best_overall_path}: {best_overall:.2f} dB at {best_overall_freq:.3f} GHz")
        
        if best_overall >= 4.9:
            print(f"\n✅✅✅ TUMOR DETECTED! (>4.9 dB threshold) ✅✅✅")
        else:
            print(f"\n⚠️  Detection below threshold")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 All plots saved in: {plots_folder}/")
    print("="*70)

if __name__ == "__main__":
    main()