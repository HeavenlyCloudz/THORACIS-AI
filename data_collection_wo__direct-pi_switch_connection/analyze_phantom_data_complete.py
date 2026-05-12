# analyze_phantom_data_complete.py
"""
PULMO AI: Complete Phantom Data Analysis
Handles: Air Baseline, Dynamic Healthy, Dynamic Tumor, Ex Vivo Healthy, Ex Vivo Tumor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from collections import defaultdict
import json

def db_to_linear(db):
    return 10 ** (db / 10)

def linear_to_db(linear):
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def load_phantom_data(folder, baseline_dict=None, verbose=False):
    """Load all CSV files from a phantom folder, apply background subtraction"""
    data_by_path = defaultdict(list)
    frequencies = None
    
    csv_files = sorted(Path(folder).glob('*.csv'))
    
    for f in csv_files:
        df = pd.read_csv(f)
        if frequencies is None:
            frequencies = df['Frequency_Hz'].values / 1e9
        
        s21_db = df['S21_dB'].values
        
        # Extract path number and rotation
        filename = f.stem
        parts = filename.split('_')
        
        path_num = None
        rotation = None
        for part in parts:
            if part.startswith('path'):
                path_num = int(part[4:6] if len(part) > 5 else part[4])
            if part.startswith('rot'):
                rotation = int(part[3:6] if len(part) > 4 else part[3])
        
        if path_num and rotation is not None:
            # Apply background subtraction
            if baseline_dict and (path_num, rotation) in baseline_dict:
                baseline_linear = db_to_linear(baseline_dict[(path_num, rotation)])
                signal_linear = db_to_linear(s21_db)
                corrected_linear = signal_linear - baseline_linear
                s21_db = linear_to_db(corrected_linear)
            
            data_by_path[(path_num, rotation)].append(s21_db)
    
    # Average multiple runs
    averaged_data = {}
    for key, s21_list in data_by_path.items():
        averaged_data[key] = np.mean(s21_list, axis=0) if len(s21_list) > 1 else s21_list[0]
    
    return averaged_data, frequencies

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_phantom_data_complete.py <data_folder>")
        return
    
    base_folder = sys.argv[1]
    
    if not os.path.exists(base_folder):
        print(f"❌ Folder not found: {base_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f" PULMO AI - COMPLETE PHANTOM ANALYSIS")
    print(f" {base_folder}")
    print(f"{'='*70}")
    
    # Load metadata
    with open(f"{base_folder}/scan_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Group phantoms by type
    phantom_groups = {
        'baseline': [],
        'dynamic_healthy': [],
        'dynamic_tumor': [],
        'exvivo_healthy': [],
        'exvivo_tumor': []
    }
    
    for phantom in metadata['phantoms']:
        ptype = phantom['phantom_type']
        if ptype in phantom_groups:
            phantom_groups[ptype].append(phantom['name'])
    
    print(f"\n📊 Phantom Groups:")
    print(f"   Baseline: {phantom_groups['baseline']}")
    print(f"   Dynamic Healthy: {len(phantom_groups['dynamic_healthy'])} phantoms")
    print(f"   Dynamic Tumor: {len(phantom_groups['dynamic_tumor'])} phantoms")
    print(f"   Ex Vivo Healthy: {len(phantom_groups['exvivo_healthy'])} phantoms")
    print(f"   Ex Vivo Tumor: {len(phantom_groups['exvivo_tumor'])} phantoms")
    
    # Load baseline first
    baseline_folder = f"{base_folder}/{phantom_groups['baseline'][0]}"
    baseline_data, frequencies = load_phantom_data(baseline_folder)
    
    # Load all phantom data with background subtraction
    all_results = {}
    
    for group_name, phantom_names in phantom_groups.items():
        if group_name == 'baseline':
            continue
        
        all_results[group_name] = []
        
        for phantom_name in phantom_names:
            phantom_folder = f"{base_folder}/{phantom_name}"
            data, _ = load_phantom_data(phantom_folder, baseline_data)
            
            if data:
                all_results[group_name].append(data)
    
    # Calculate group averages
    print("\n📊 CALCULATING GROUP STATISTICS...")
    
    group_stats = {}
    for group_name, results_list in all_results.items():
        if not results_list:
            continue
        
        # Collect all S21 values across all phantoms in this group
        all_s21 = []
        for data_dict in results_list:
            for (path, rot), s21 in data_dict.items():
                all_s21.extend(s21)
        
        all_s21 = np.array(all_s21)
        group_stats[group_name] = {
            'mean': np.mean(all_s21),
            'std': np.std(all_s21),
            'min': np.min(all_s21),
            'max': np.max(all_s21),
            'num_scans': len(all_s21),
            'num_phantoms': len(results_list)
        }
        
        print(f"\n   {group_name.upper()}:")
        print(f"      Mean S21: {group_stats[group_name]['mean']:.2f} ± {group_stats[group_name]['std']:.2f} dB")
        print(f"      Range: {group_stats[group_name]['min']:.2f} to {group_stats[group_name]['max']:.2f} dB")
        print(f"      Phantoms: {group_stats[group_name]['num_phantoms']}")
    
    # Tumor detection analysis
    print(f"\n{'='*70}")
    print(f"🎯 TUMOR DETECTION ANALYSIS")
    print(f"{'='*70}")
    
    if 'dynamic_healthy' in group_stats and 'dynamic_tumor' in group_stats:
        healthy_mean = group_stats['dynamic_healthy']['mean']
        tumor_mean = group_stats['dynamic_tumor']['mean']
        drop = healthy_mean - tumor_mean
        
        print(f"\n📊 DYNAMIC AGAR PHANTOMS:")
        print(f"   Healthy Mean: {healthy_mean:.2f} dB")
        print(f"   Tumor Mean: {tumor_mean:.2f} dB")
        print(f"   Signal Drop: {drop:.2f} dB")
        
        if drop >= 4.9:
            print(f"   ✅✅✅ TUMOR DETECTED! ({drop:.2f} dB > 4.9 dB)")
        else:
            print(f"   ⚠️ Detection below threshold")
    
    if 'exvivo_healthy' in group_stats and 'exvivo_tumor' in group_stats:
        healthy_mean = group_stats['exvivo_healthy']['mean']
        tumor_mean = group_stats['exvivo_tumor']['mean']
        drop = healthy_mean - tumor_mean
        
        print(f"\n📊 EX VIVO CHICKEN PHANTOMS:")
        print(f"   Healthy Mean: {healthy_mean:.2f} dB")
        print(f"   Tumor Mean: {tumor_mean:.2f} dB")
        print(f"   Signal Drop: {drop:.2f} dB")
        
        if drop >= 4.9:
            print(f"   ✅✅✅ TUMOR DETECTED! ({drop:.2f} dB > 4.9 dB)")
        else:
            print(f"   ⚠️ Detection below threshold")
    
    # Comparison plot
    plt.figure(figsize=(12, 6))
    
    groups_to_plot = ['dynamic_healthy', 'dynamic_tumor', 'exvivo_healthy', 'exvivo_tumor']
    colors = ['green', 'red', 'lightgreen', 'salmon']
    labels = ['Dynamic Healthy', 'Dynamic Tumor', 'Ex Vivo Healthy', 'Ex Vivo Tumor']
    
    for i, group in enumerate(groups_to_plot):
        if group in group_stats:
            stats = group_stats[group]
            plt.bar(i, stats['mean'], yerr=stats['std'], color=colors[i], 
                   capsize=5, label=labels[i])
    
    plt.axhline(y=-29.47, color='blue', linestyle='--', label='Air Baseline')
    plt.xticks(range(4), labels, rotation=15)
    plt.ylabel('S21 (dB)')
    plt.title('PULMO AI: All Phantom Types Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_folder}/all_phantoms_comparison.png", dpi=150)
    plt.show()
    
    print(f"\n📁 Analysis saved in: {base_folder}/")
    print(f"   - all_phantoms_comparison.png")

if __name__ == "__main__":
    main()
