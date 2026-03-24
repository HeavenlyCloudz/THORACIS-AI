# background_subtraction_test.py
"""
Quick test to remove direct coupling using background subtraction
Run this after scanning:
- Air baseline (no phantom) for Path 1
- Tumor phantom for Path 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_s21_csv(filepath):
    """Load S21 data from CSV file"""
    df = pd.read_csv(filepath)
    frequencies = df['Frequency_Hz'].values / 1e9  # Convert to GHz
    s21_db = df['S21_dB'].values
    return frequencies, s21_db

def db_to_linear(db):
    """Convert dB to linear magnitude"""
    return 10 ** (db / 10)

def linear_to_db(linear):
    """Convert linear magnitude to dB"""
    # Add small epsilon to avoid log(0)
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def background_subtract(air_s21_db, phantom_s21_db):
    """
    Subtract air baseline (coupling) from phantom scan
    This removes direct antenna coupling that didn't go through phantom
    """
    # Convert to linear power
    air_linear = db_to_linear(air_s21_db)
    phantom_linear = db_to_linear(phantom_s21_db)
    
    # Subtract coupling (assuming coupling is additive in power domain)
    corrected_linear = phantom_linear - air_linear
    
    # Convert back to dB
    corrected_db = linear_to_db(corrected_linear)
    
    return corrected_db

def main():
    print("="*70)
    print(" PULMO AI - BACKGROUND SUBTRACTION TEST")
    print("="*70)
    
    # =========================================================
    # CONFIGURE THESE PATHS
    # =========================================================
    
    # Path to your air baseline file (no phantom)
    air_file = input("Enter path to Air Baseline CSV: ").strip()
    
    # Path to your tumor phantom file
    tumor_file = input("Enter path to Tumor Phantom CSV: ").strip()
    
    # Optional: healthy phantom file
    healthy_file = input("Enter path to Healthy Phantom CSV (or press Enter to skip): ").strip()
    
    # =========================================================
    # LOAD DATA
    # =========================================================
    
    print("\n📂 Loading data...")
    
    freq_air, air_s21 = load_s21_csv(air_file)
    freq_tumor, tumor_s21 = load_s21_csv(tumor_file)
    
    print(f"   Air baseline: {len(air_s21)} points, mean={np.mean(air_s21):.2f} dB")
    print(f"   Tumor phantom: {len(tumor_s21)} points, mean={np.mean(tumor_s21):.2f} dB")
    
    if healthy_file:
        freq_healthy, healthy_s21 = load_s21_csv(healthy_file)
        print(f"   Healthy phantom: {len(healthy_s21)} points, mean={np.mean(healthy_s21):.2f} dB")
    
    # =========================================================
    # APPLY BACKGROUND SUBTRACTION
    # =========================================================
    
    print("\n🔧 Applying background subtraction...")
    
    tumor_corrected = background_subtract(air_s21, tumor_s21)
    
    print(f"   Tumor after subtraction: mean={np.mean(tumor_corrected):.2f} dB")
    
    if healthy_file:
        healthy_corrected = background_subtract(air_s21, healthy_s21)
        print(f"   Healthy after subtraction: mean={np.mean(healthy_corrected):.2f} dB")
    
    # =========================================================
    # ANALYSIS - FIND BEST DETECTION FREQUENCY
    # =========================================================
    
    print("\n📊 Analyzing tumor detection...")
    
    if healthy_file:
        # Compare healthy vs tumor after subtraction
        diff = healthy_corrected - tumor_corrected
        best_idx = np.argmax(diff)
        best_freq = freq_air[best_idx]
        best_drop = diff[best_idx]
        
        print(f"\n🎯 TUMOR DETECTION (Healthy vs Tumor):")
        print(f"   Best frequency: {best_freq:.3f} GHz")
        print(f"   Signal drop: {best_drop:.2f} dB")
        
        if best_drop >= 4.9:
            print(f"\n✅✅✅ TUMOR DETECTED! ({best_drop:.2f} dB > 4.9 dB threshold) ✅✅✅")
        elif best_drop >= 2.0:
            print(f"\n✅ WEAK DETECTION ({best_drop:.2f} dB) - System works, improve phantom")
        else:
            print(f"\n⚠️  No detection ({best_drop:.2f} dB) - Increase phantom contrast")
    
    # Compare tumor vs air (should show large difference)
    tumor_vs_air = tumor_corrected - air_s21
    best_idx = np.argmax(tumor_vs_air)
    best_freq = freq_air[best_idx]
    best_diff = tumor_vs_air[best_idx]
    
    print(f"\n📡 Tumor vs Air (after subtraction):")
    print(f"   Best frequency: {best_freq:.3f} GHz")
    print(f"   Difference: {best_diff:.2f} dB")
    
    # =========================================================
    # PLOT RESULTS
    # =========================================================
    
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Raw data (before subtraction)
    ax = axes[0, 0]
    ax.plot(freq_air, air_s21, label='Air (coupling)', color='blue', alpha=0.7)
    ax.plot(freq_tumor, tumor_s21, label='Tumor (raw)', color='red', alpha=0.7)
    if healthy_file:
        ax.plot(freq_healthy, healthy_s21, label='Healthy (raw)', color='green', alpha=0.7)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S21 (dB)')
    ax.set_title('Raw Data (Before Subtraction)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-60, -20)
    
    # Plot 2: After background subtraction
    ax = axes[0, 1]
    ax.plot(freq_air, air_s21, label='Air (coupling)', color='blue', alpha=0.5, linestyle='--')
    ax.plot(freq_tumor, tumor_corrected, label='Tumor (corrected)', color='red', linewidth=2)
    if healthy_file:
        ax.plot(freq_healthy, healthy_corrected, label='Healthy (corrected)', color='green', linewidth=2)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S21 (dB)')
    ax.set_title('After Background Subtraction (Coupling Removed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-60, -20)
    
    # Plot 3: Difference (Healthy vs Tumor)
    if healthy_file:
        ax = axes[1, 0]
        diff = healthy_corrected - tumor_corrected
        ax.plot(freq_air, diff, color='purple', linewidth=2)
        ax.axhline(y=4.9, color='red', linestyle='--', label='Detection Threshold (4.9 dB)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.fill_between(freq_air, 0, diff, where=(diff > 0), color='green', alpha=0.3, label='Tumor detected')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Signal Drop (dB)')
        ax.set_title('Tumor Detection Signal (Healthy - Tumor)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annotate best frequency
        best_idx = np.argmax(diff)
        ax.annotate(f'Peak: {diff[best_idx]:.1f} dB\n@ {freq_air[best_idx]:.2f} GHz',
                   xy=(freq_air[best_idx], diff[best_idx]),
                   xytext=(freq_air[best_idx] + 0.1, diff[best_idx] + 2),
                   arrowprops=dict(arrowstyle='->', color='black'))
    
    # Plot 4: Frequency response comparison
    ax = axes[1, 1]
    ax.plot(freq_air, tumor_corrected, label='Tumor (corrected)', color='red', linewidth=2)
    if healthy_file:
        ax.plot(freq_healthy, healthy_corrected, label='Healthy (corrected)', color='green', linewidth=2)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('S21 (dB)')
    ax.set_title('Comparison After Subtraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-60, -20)
    
    plt.tight_layout()
    plt.savefig('background_subtraction_results.png', dpi=150)
    print("   ✅ Saved: background_subtraction_results.png")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print(f"\n📊 Raw Data:")
    print(f"   Air baseline:  {np.mean(air_s21):.2f} dB")
    print(f"   Tumor phantom: {np.mean(tumor_s21):.2f} dB")
    print(f"   Difference:    {np.mean(tumor_s21) - np.mean(air_s21):.2f} dB")
    
    print(f"\n🔧 After Background Subtraction:")
    print(f"   Tumor corrected: {np.mean(tumor_corrected):.2f} dB")
    
    if healthy_file:
        print(f"   Healthy corrected: {np.mean(healthy_corrected):.2f} dB")
        print(f"   Tumor - Healthy:   {np.mean(tumor_corrected) - np.mean(healthy_corrected):.2f} dB")
        print(f"\n🎯 Best detection: {best_drop:.2f} dB at {best_freq:.3f} GHz")
        
        if best_drop >= 4.9:
            print("\n✅✅✅ SUCCESS! Tumor detected above 4.9 dB threshold")
        elif best_drop >= 2.0:
            print("\n✅ PARTIAL SUCCESS - System detects contrast, but phantom needs improvement")
        else:
            print("\n⚠️  No detection - Increase phantom contrast or reduce antenna gap")
    
    print("\n" + "="*70)
    plt.show()

if __name__ == "__main__":
    main()