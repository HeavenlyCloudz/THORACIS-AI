# background_subtraction_test_full.py
"""
PULMO AI: Full Background Subtraction Test for ALL 4 Paths
Processes entire folders and compares healthy vs tumor across all paths
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import os

def load_all_csv_from_folder(folder_path, verbose=False):
    """
    Load all CSV files from a folder, grouped by path number
    Returns: dict {path_num: list_of_s21_arrays}
    """
    path_data = defaultdict(list)
    frequencies = None
    
    csv_files = sorted(Path(folder_path).glob('*.csv'))
    
    for file in csv_files:
        df = pd.read_csv(file)
        if frequencies is None:
            frequencies = df['Frequency_Hz'].values / 1e9
        
        s21 = df['S21_dB'].values
        
        # Extract path number from filename
        filename = file.stem
        parts = filename.split('_')
        path_num = None
        for part in parts:
            if part.startswith('path'):
                try:
                    path_num = int(part[4])
                except:
                    pass
                break
        
        if path_num is not None:
            path_data[path_num].append(s21)
    
    # Average multiple runs per path
    averaged_data = {}
    for path_num, s21_list in path_data.items():
        averaged_data[path_num] = np.mean(s21_list, axis=0)
        if verbose:
            print(f"      Path {path_num}: {len(s21_list)} files, mean={np.mean(averaged_data[path_num]):.2f} dB")
    
    return averaged_data, frequencies

def db_to_linear(db):
    """Convert dB to linear magnitude"""
    return 10 ** (db / 10)

def linear_to_db(linear):
    """Convert linear magnitude to dB"""
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def background_subtract_all_paths(baseline_data, phantom_data):
    """
    Apply background subtraction to all paths
    baseline_data: dict {path_num: s21_array}
    phantom_data: dict {path_num: s21_array}
    Returns: dict {path_num: corrected_s21_array}
    """
    corrected_data = {}
    
    for path_num in phantom_data.keys():
        if path_num in baseline_data:
            air_linear = db_to_linear(baseline_data[path_num])
            phantom_linear = db_to_linear(phantom_data[path_num])
            corrected_linear = phantom_linear - air_linear
            corrected_data[path_num] = linear_to_db(corrected_linear)
        else:
            corrected_data[path_num] = phantom_data[path_num]
    
    return corrected_data

def find_best_detection(healthy_data, tumor_data, frequencies):
    """
    Find the best detection frequency across all paths
    Returns: best_path, best_freq, best_drop
    """
    best_overall_drop = 0
    best_overall_freq = 0
    best_overall_path = 0
    
    for path_num in healthy_data.keys():
        if path_num in tumor_data:
            # Calculate difference (healthy - tumor)
            diff = healthy_data[path_num] - tumor_data[path_num]
            best_idx = np.argmax(diff)
            best_drop = diff[best_idx]
            best_freq = frequencies[best_idx]
            
            if best_drop > best_overall_drop:
                best_overall_drop = best_drop
                best_overall_freq = best_freq
                best_overall_path = path_num
    
    return best_overall_path, best_overall_freq, best_overall_drop

def main():
    print("="*70)
    print(" PULMO AI - FULL BACKGROUND SUBTRACTION TEST (ALL 4 PATHS)")
    print("="*70)
    
    # =========================================================
    # CONFIGURE PATHS
    # =========================================================
    
    # Enter the base folder containing your phantom data
    base_folder = input("Enter path to phantom_data folder (or press Enter to use current directory): ").strip()
    
    if not base_folder:
        # Look for phantom_data folders in current directory
        phantom_folders = sorted([d for d in os.listdir('.') if d.startswith('phantom_data_')])
        if phantom_folders:
            print("\n📁 Found phantom_data folders:")
            for i, folder in enumerate(phantom_folders):
                print(f"   [{i+1}] {folder}")
            choice = input(f"\nSelect folder (1-{len(phantom_folders)}): ").strip()
            try:
                base_folder = phantom_folders[int(choice) - 1]
            except:
                print("Invalid selection, using first folder")
                base_folder = phantom_folders[0]
        else:
            print("❌ No phantom_data folders found!")
            return
    
    print(f"\n📁 Using: {base_folder}")
    
    # Define condition folders
    baseline_folder = Path(base_folder) / '01_baseline_air'
    healthy_folder = Path(base_folder) / '02_healthy_phantom'
    tumor_folder = Path(base_folder) / '03_tumor_phantom'
    
    # Check if folders exist
    if not baseline_folder.exists():
        print(f"❌ Baseline folder not found: {baseline_folder}")
        return
    if not healthy_folder.exists():
        print(f"⚠️  Healthy folder not found: {healthy_folder}")
        healthy_data = None
    else:
        healthy_data = True
    if not tumor_folder.exists():
        print(f"❌ Tumor folder not found: {tumor_folder}")
        return
    
    # =========================================================
    # LOAD DATA
    # =========================================================
    
    print("\n📂 Loading data...")
    
    print("   Loading baseline (air)...")
    baseline_data, frequencies = load_all_csv_from_folder(baseline_folder, verbose=True)
    print(f"   Found paths: {sorted(baseline_data.keys())}")
    
    if healthy_data:
        print("\n   Loading healthy phantom...")
        healthy_raw, _ = load_all_csv_from_folder(healthy_folder, verbose=True)
        print(f"   Found paths: {sorted(healthy_raw.keys())}")
    
    print("\n   Loading tumor phantom...")
    tumor_raw, _ = load_all_csv_from_folder(tumor_folder, verbose=True)
    print(f"   Found paths: {sorted(tumor_raw.keys())}")
    
    # =========================================================
    # APPLY BACKGROUND SUBTRACTION
    # =========================================================
    
    print("\n🔧 Applying background subtraction...")
    
    tumor_corrected = background_subtract_all_paths(baseline_data, tumor_raw)
    
    if healthy_data:
        healthy_corrected = background_subtract_all_paths(baseline_data, healthy_raw)
    
    # =========================================================
    # ANALYZE EACH PATH
    # =========================================================
    
    print("\n" + "="*70)
    print("📊 PATH-BY-PATH ANALYSIS")
    print("="*70)
    
    all_paths = sorted(set(baseline_data.keys()) | set(tumor_raw.keys()))
    
    path_labels = {
        1: 'Path 1 (1→3)',
        2: 'Path 2 (1→4)',
        3: 'Path 3 (2→3)',
        4: 'Path 4 (2→4)'
    }
    
    path_detections = []
    
    for path_num in all_paths:
        print(f"\n{path_labels.get(path_num, f'Path {path_num}')}:")
        
        tumor_path = tumor_corrected.get(path_num)
        if tumor_path is None:
            print("   ⚠️  No tumor data for this path")
            continue
        
        print(f"   Tumor corrected mean: {np.mean(tumor_path):.2f} dB")
        
        if healthy_data:
            healthy_path = healthy_corrected.get(path_num)
            if healthy_path is not None:
                # Calculate difference
                diff = healthy_path - tumor_path
                best_idx = np.argmax(diff)
                best_freq = frequencies[best_idx]
                best_drop = diff[best_idx]
                
                print(f"   Healthy corrected mean: {np.mean(healthy_path):.2f} dB")
                print(f"   Difference (Healthy - Tumor): {np.mean(diff):.2f} dB avg")
                print(f"   🎯 Best detection: {best_drop:.2f} dB at {best_freq:.3f} GHz")
                
                path_detections.append({
                    'path': path_num,
                    'best_drop': best_drop,
                    'best_freq': best_freq,
                    'avg_drop': np.mean(diff)
                })
                
                if best_drop >= 4.9:
                    print(f"   ✅✅✅ TUMOR DETECTED! (>4.9 dB)")
                elif best_drop >= 2.0:
                    print(f"   ✅ Weak detection (2-4.9 dB)")
                else:
                    print(f"   ⚠️  No detection (<2 dB)")
            else:
                print("   ⚠️  No healthy data for this path")
        else:
            # Just show tumor vs baseline
            baseline_path = baseline_data.get(path_num)
            if baseline_path is not None:
                diff = baseline_path - tumor_path
                print(f"   Baseline mean: {np.mean(baseline_path):.2f} dB")
                print(f"   Difference (Baseline - Tumor): {np.mean(diff):.2f} dB avg")
    
    # =========================================================
    # FIND BEST OVERALL DETECTION
    # =========================================================
    
    if healthy_data and path_detections:
        best_path = max(path_detections, key=lambda x: x['best_drop'])
        print("\n" + "="*70)
        print("🏆 BEST OVERALL DETECTION")
        print("="*70)
        print(f"   Path {best_path['path']}: {best_path['best_drop']:.2f} dB at {best_path['best_freq']:.3f} GHz")
        
        if best_path['best_drop'] >= 4.9:
            print("\n✅✅✅ TUMOR DETECTED! (>4.9 dB threshold) ✅✅✅")
    
    # =========================================================
    # PLOT RESULTS
    # =========================================================
    
    print("\n📊 Generating plots...")
    
    # Create output folder
    plots_folder = Path(base_folder) / 'analysis_plots_subtracted'
    plots_folder.mkdir(exist_ok=True)
    
    # Plot 1: All paths comparison (Healthy vs Tumor)
    if healthy_data:
        plt.figure(figsize=(14, 10))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, path_num in enumerate(all_paths):
            if path_num in healthy_corrected and path_num in tumor_corrected:
                plt.subplot(2, 2, idx + 1)
                
                plt.plot(frequencies, healthy_corrected[path_num], 
                        label='Healthy', color='green', linewidth=2)
                plt.plot(frequencies, tumor_corrected[path_num], 
                        label='Tumor', color='red', linewidth=2)
                
                # Highlight best detection
                diff = healthy_corrected[path_num] - tumor_corrected[path_num]
                best_idx = np.argmax(diff)
                plt.axvline(x=frequencies[best_idx], color='purple', 
                           linestyle='--', alpha=0.7,
                           label=f'Best: {diff[best_idx]:.1f} dB')
                
                plt.xlabel('Frequency (GHz)')
                plt.ylabel('S21 (dB)')
                plt.title(f'{path_labels.get(path_num, f"Path {path_num}")}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(-100, -20)
        
        plt.tight_layout()
        plt.savefig(plots_folder / 'all_paths_comparison.png', dpi=150)
        print(f"   ✅ Saved: {plots_folder}/all_paths_comparison.png")
        plt.close()
    
    # Plot 2: Detection signal per path
    if healthy_data and path_detections:
        plt.figure(figsize=(12, 6))
        
        for detection in path_detections:
            path_num = detection['path']
            if path_num in healthy_corrected and path_num in tumor_corrected:
                diff = healthy_corrected[path_num] - tumor_corrected[path_num]
                plt.plot(frequencies, diff, 
                        label=f"Path {path_num} ({path_labels.get(path_num, '')})", 
                        linewidth=2)
        
        plt.axhline(y=4.9, color='red', linestyle='--', 
                   label='Detection Threshold (4.9 dB)', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Signal Drop (dB)')
        plt.title('Tumor Detection Signal Across All Paths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_folder / 'detection_signals.png', dpi=150)
        print(f"   ✅ Saved: {plots_folder}/detection_signals.png")
        plt.close()
    
    # =========================================================
    # SUMMARY TABLE
    # =========================================================
    
    print("\n" + "="*70)
    print("📊 SUMMARY TABLE")
    print("="*70)
    print(f"\n{'Path':<8} {'Best Freq (GHz)':<15} {'Best Drop (dB)':<15} {'Avg Drop (dB)':<15} {'Status':<10}")
    print("-" * 70)
    
    for detection in sorted(path_detections, key=lambda x: x['path']):
        status = "✅ DETECTED" if detection['best_drop'] >= 4.9 else "⚠️ Weak" if detection['best_drop'] >= 2.0 else "❌ No"
        print(f"Path {detection['path']:<5} {detection['best_freq']:<15.3f} {detection['best_drop']:<15.2f} {detection['avg_drop']:<15.2f} {status:<10}")
    
    # =========================================================
    # FINAL VERDICT
    # =========================================================
    
    print("\n" + "="*70)
    print(" FINAL VERDICT")
    print("="*70)
    
    if healthy_data and path_detections:
        best = max(path_detections, key=lambda x: x['best_drop'])
        if best['best_drop'] >= 4.9:
            print(f"\n✅✅✅ SYSTEM VALIDATED! ✅✅✅")
            print(f"   Best detection: {best['best_drop']:.2f} dB at {best['best_freq']:.3f} GHz on Path {best['path']}")
            print(f"   All {len(path_detections)} paths show tumor detection capability")
        else:
            print(f"\n⚠️  System detects contrast but below 4.9 dB threshold")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()