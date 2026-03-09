import serial
import time
import math

PORT = 'COM3'
BAUDRATE = 115200

with serial.Serial(PORT, BAUDRATE, timeout=2) as ser:
    time.sleep(2)
    ser.reset_input_buffer()
    
    # Send scan command
    ser.write(b'scan 2000000000 3000000000 11 5\n')  # Only 11 points for speed
    time.sleep(1)
    
    # Read and parse 5 points
    for _ in range(5):
        line = ser.readline().decode('ascii', errors='ignore').strip()
        if line and not line.startswith(('ch>', 'scan')):
            parts = line.split()
            if len(parts) >= 3:
                freq = float(parts[0])
                real = float(parts[1])
                imag = float(parts[2])
                mag_db = 20 * math.log10((real**2 + imag**2)**0.5)
                print(f"{freq/1e9:.3f} GHz: {real:.3f} + j{imag:.3f} = {mag_db:.1f} dB")