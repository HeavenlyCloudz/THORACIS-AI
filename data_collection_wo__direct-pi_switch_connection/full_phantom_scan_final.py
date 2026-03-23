# full_phantom_scan_final.py
"""
PULMO AI - FINAL SCAN SCRIPT
Optimized for your working frequencies (106 good points)
"""
import serial
import time
import csv
import math
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# CONFIGURATION
VNA_PORT = 'COM4'
BAUDRATE = 115200
START_FREQ = 2000000000
STOP_FREQ = 3000000000
POINTS = 201
NUM_AVERAGES = 5  # More averaging for better SNR

# Your best frequencies from the diagnostic
BEST_FREQS = [2.10, 2.20, 2.80, 2.90]  # GHz - where signal is strongest

def capture_with_averaging():
    """Capture with averaging for better SNR"""
    all_sweeps = []
    
    for avg in range(NUM_AVERAGES):
        with serial.Serial(VNA_PORT, BAUDRATE, timeout=2) as ser:
            time.sleep(1)
            ser.reset_input_buffer()
            
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5"
            ser.write((cmd + '\n').encode())
            time.sleep(2.5)  # Longer sweep = better SNR
            
            sweep_data = []
            for _ in range(POINTS):
                line = ser.readline().decode().strip()
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        freq = float(parts[0])
                        real = float(parts[1])
                        imag = float(parts[2])
                        mag = math.sqrt(real**2 + imag**2)
                        db = 20 * math.log10(mag) if mag > 0 else -120
                        # Only keep reasonable values
                        if db > -100:  # Filter out garbage readings
                            sweep_data.append([freq, db])
                    except:
                        continue
            
            if len(sweep_data) > 100:  # At least 100 good points
                all_sweeps.append(sweep_data)
                print(f"      Sweep {avg+1}/{NUM_AVERAGES}: {len(sweep_data)} points", end='')
                if avg < NUM_AVERAGES - 1:
                    print()
    
    if all_sweeps:
        # Interpolate to common frequency grid
        freqs = np.linspace(START_FREQ, STOP_FREQ, POINTS)
        avg_data = np.zeros(POINTS)
        
        for sweep in all_sweeps:
            sweep_freqs = [d[0] for d in sweep]
            sweep_dbs = [d[1] for d in sweep]
            # Interpolate to grid
            interp_db = np.interp(freqs, sweep_freqs, sweep_dbs)
            avg_data += interp_db
        
        avg_data /= len(all_sweeps)
        
        # Return as list of [freq, db]
        return [[freqs[i], avg_data[i]] for i in range(POINTS)]
    
    return None

def analyze_peak_frequencies(data):
    """Find where your signal is strongest"""
    freqs = [d[0]/1e9 for d in data]
    dbs = [d[1] for d in data]
    
    # Find peaks above -30 dB
    peaks = []
    for i in range(len(dbs)):
        if dbs[i] > -30:
            peaks.append(freqs[i])
    
    return peaks

def main():
    print("="*60)
    print("PULMO AI - FINAL SCAN (Optimized for Your Setup)")
    print("="*60)
    
    print("\n✨ REMINDER: You have 106 usable frequencies!")
    print("   That's MORE than enough for tumor detection")
    print("   Focus on RELATIVE changes, not absolute values\n")
    
    # Test conditions (ONLY 3 - quick and effective)
    CONDITIONS = [
        {'name': '01_baseline_air', 'prompt': 'Remove ALL phantoms'},
        {'name': '02_healthy_phantom', 'prompt': 'Place HEALTHY phantom (agar only)'},
        {'name': '03_tumor_phantom', 'prompt': 'Place TUMOR phantom (with aluminum foil ball)'}
    ]
    
    ROTATIONS = [0, 120, 240]  # 3 rotations for spatial data
    PATHS = [
        {'num': 1, 'name': '1→3'},
        {'num': 2, 'name': '1→4'},
        {'num': 3, 'name': '2→3'},
        {'num': 4, 'name': '2→4'}
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"phantom_data_{timestamp}"
    os.makedirs(base_folder, exist_ok=True)
    
    print(f"📁 Data folder: {base_folder}")
    print(f"📊 Total files: {len(CONDITIONS)} × {len(ROTATIONS)} × {len(PATHS)} = {len(CONDITIONS)*len(ROTATIONS)*len(PATHS)}")
    print(f"⏱️  Estimated time: ~15-20 minutes\n")
    
    print("🔧 On Pi terminal, run: python pi_switch_controller.py")
    print("   Then follow prompts here\n")
    
    all_results = []
    
    for cond_idx, condition in enumerate(CONDITIONS, 1):
        print(f"\n{'='*60}")
        print(f"📋 CONDITION {cond_idx}/3: {condition['name']}")
        print(f"👉 {condition['prompt']}")
        print(f"{'='*60}")
        
        cond_folder = f"{base_folder}/{condition['name']}"
        os.makedirs(cond_folder, exist_ok=True)
        
        input("\nPress ENTER when ready...")
        
        for rot_idx, rotation in enumerate(ROTATIONS, 1):
            print(f"\n  🔄 Rotation {rot_idx}/3: {rotation}°")
            
            for path in PATHS:
                print(f"\n    🔧 Set Pi to Path {path['num']} ({path['name']})")
                print(f"       → On Pi terminal, type: {path['num']}")
                input("       Press ENTER after setting path...")
                
                print(f"    📡 Capturing ({NUM_AVERAGES} sweeps with averaging)...", end='', flush=True)
                data = capture_with_averaging()
                
                if data:
                    # Save data
                    filename = f"path{path['num']}_rot{rotation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    full_path = os.path.join(cond_folder, filename)
                    
                    with open(full_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Frequency_Hz', 'S21_dB'])
                        writer.writerows(data)
                    
                    # Analyze peaks
                    peaks = analyze_peak_frequencies(data)
                    s21_vals = [d[1] for d in data]
                    avg_db = np.mean([d for d in s21_vals if d > -100])  # Ignore noise
                    
                    print(f" ✅ avg={avg_db:.1f}dB, {len(peaks)} usable frequencies")
                    
                    # Store for comparison
                    all_results.append({
                        'condition': condition['name'],
                        'rotation': rotation,
                        'path': path['num'],
                        'avg_db': avg_db,
                        'peaks': peaks
                    })
                else:
                    print(" ❌ Failed - check connections")
        
        print(f"\n✅ {condition['name']} complete!")
    
    # Save summary
    import json
    with open(f"{base_folder}/scan_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ SCANNING COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Data saved: {base_folder}")
    
    # Calculate averages per condition
    print("\n📊 QUICK SUMMARY:")
    for condition in CONDITIONS:
        cond_results = [r for r in all_results if r['condition'] == condition['name']]
        if cond_results:
            avg = np.mean([r['avg_db'] for r in cond_results])
            print(f"   {condition['name']}: {avg:.1f} dB average")
    
    print("\n🔍 NEXT: Run analysis to see tumor detection")
    print(f"python analyze_results.py {base_folder}")
    print("\n🎉 You did it! Your data is ready for ML!")

if __name__ == "__main__":
    main()
