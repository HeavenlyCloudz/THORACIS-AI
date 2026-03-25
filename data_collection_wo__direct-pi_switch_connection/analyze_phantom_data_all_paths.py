# analyze_phantom_data_all_paths.py
"""
Analyzes all phantom data with BACKGROUND SUBTRACTION
Now uses ALL 4 paths (after subtraction, all work)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from collections import defaultdict

# CONFIGURATION - Use ALL 4 paths
VALID_PATHS = [1, 2, 3, 4]
PATH_LABELS = {
    1: 'Path 1 (1→3) - Opposite',
    2: 'Path 2 (1→4) - Diagonal',
    3: 'Path 3 (2→3) - Diagonal',
    4: 'Path 4 (2→4) - Opposite',
}
PATH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def db_to_linear(db):
    return 10 ** (db / 10)

def linear_to_db(linear):
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def load_condition_data_with_background(folder, baseline_data_dict=None, verbose=False):
    """Load CSV files, apply background subtraction if baseline provided"""
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
        
        if path_num and rotation is not None and path_num in VALID_PATHS:
            if baseline_data_dict and (path_num, rotation) in baseline_data_dict:
                baseline_linear = db_to_linear(baseline_data_dict[(path_num, rotation)])
                signal_linear = db_to_linear(s21_db)
                corrected_linear = signal_linear - baseline_linear
                s21_db = linear_to_db(corrected_linear)
            
            data_by_path_rotation[(path_num, rotation)].append(s21_db)
    
    averaged_data = {}
    for key, s21_list in data_by_path_rotation.items():
        averaged_data[key] = np.mean(s21_list, axis=0) if len(s21_list) > 1 else s21_list[0]
    
    if verbose:
        print(f"    Loaded {len(averaged_data)} (path, rotation) combinations")
    
    return averaged_data, frequencies

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_phantom_data_all_paths.py <data_folder>")
        return
    
    base_folder = sys.argv[1]
    
    if not os.path.exists(base_folder):
        print(f"❌ Folder not found: {base_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f" PULMO AI - PHANTOM DATA ANALYSIS (ALL 4 PATHS)")
    print(f" {base_folder}")
    print(f"{'='*70}\n")
    print(f"📡 USING ALL PATHS {VALID_PATHS}")
    
    conditions = [
        ('01_baseline_air', 'Air Baseline', '#1f77b4'),
        ('02_healthy_phantom', 'Healthy Phantom', '#2ca02c'),
        ('03_tumor_phantom', 'Tumor Phantom', '#d62728')
    ]
    
    # Load baseline (air) first
    print("\n📂 Loading Air Baseline...")
    baseline_folder = os.path.join(base_folder, '01_baseline_air')
    baseline_data, frequencies = load_condition_data_with_background(
        baseline_folder, baseline_data_dict=None, verbose=True
    )
    
    if not baseline_data:
        print("❌ No baseline data found!")
        return
    
    # Load all conditions with background subtraction
    all_data = {}
    rotations_found = set()
    
    print("\n📂 Loading conditions...")
    for folder_name, display_name, color in conditions:
        folder = os.path.join(base_folder, folder_name)
        if os.path.exists(folder):
            print(f"   {display_name}...")
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
                for (_, rotation) in data.keys():
                    rotations_found.add(rotation)
    
    rotations = sorted(rotations_found)
    print(f"\n📐 Rotations detected: {rotations}")
    
    # Create output folder
    plots_folder = os.path.join(base_folder, 'analysis_plots_all_paths')
    os.makedirs(plots_folder, exist_ok=True)
    
    print("\n📊 Generating plots...")
    
    # =========================================================
    # PLOT: All 4 paths comparison (Tumor vs Healthy)
    # =========================================================
    if '02_healthy_phantom' in all_data and '03_tumor_phantom' in all_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        healthy_data = all_data['02_healthy_phantom']['data']
        tumor_data = all_data['03_tumor_phantom']['data']
        
        for idx, path_num in enumerate(VALID_PATHS):
            ax = axes[idx // 2, idx % 2]
            
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
                diff = healthy_avg - tumor_avg
                
                ax.plot(frequencies, healthy_avg, color=PATH_COLORS[path_num-1],
                       label='Healthy', linewidth=2, linestyle='--')
                ax.plot(frequencies, tumor_avg, color=PATH_COLORS[path_num-1],
                       label='Tumor', linewidth=2)
                
                # Find best detection
                best_idx = np.argmax(diff)
                best_freq = frequencies[best_idx]
                best_drop = diff[best_idx]
                
                ax.axvline(x=best_freq, color='purple', linestyle=':', alpha=0.7,
                          label=f'Peak: {best_drop:.1f}dB @ {best_freq:.2f}GHz')
                
                ax.set_title(f'{PATH_LABELS[path_num]}')
                ax.set_xlabel('Frequency (GHz)')
                ax.set_ylabel('S21 (dB)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-120, -20)
        
        plt.suptitle('PULMO AI: All 4 Paths - Healthy vs Tumor (After Background Subtraction)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, 'all_paths_comparison.png'), dpi=150)
        plt.close()
        print(f"   ✅ all_paths_comparison.png")
    
    # =========================================================
    # STATISTICS SUMMARY
    # =========================================================
    print("\n" + "="*70)
    print("📊 STATISTICS SUMMARY (ALL 4 PATHS)")
    print("="*70)
    
    if '02_healthy_phantom' in all_data and '03_tumor_phantom' in all_data:
        healthy_data = all_data['02_healthy_phantom']['data']
        tumor_data = all_data['03_tumor_phantom']['data']
        
        print(f"\n🎯 BEST DETECTION BY PATH:")
        best_overall = 0
        best_overall_path = 0
        best_overall_freq = 0
        
        for path_num in VALID_PATHS:
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
                diff = healthy_avg - tumor_avg
                best_idx = np.argmax(diff)
                best_drop = diff[best_idx]
                best_freq = frequencies[best_idx]
                
                print(f"\n   {PATH_LABELS[path_num]}:")
                print(f"      Best frequency: {best_freq:.3f} GHz")
                print(f"      Signal drop: {best_drop:.2f} dB")
                
                if best_drop > best_overall:
                    best_overall = best_drop
                    best_overall_path = path_num
                    best_overall_freq = best_freq
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST OVERALL DETECTION:")
        print(f"   {PATH_LABELS[best_overall_path]}: {best_overall:.2f} dB at {best_overall_freq:.3f} GHz")
        
        if best_overall >= 4.9:
            print(f"\n✅✅✅ TUMOR DETECTED! (>4.9 dB threshold) ✅✅✅")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()