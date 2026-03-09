import serial
import time

PORT = 'COM3'
BAUDRATE = 115200

with serial.Serial(PORT, BAUDRATE, timeout=2) as ser:
    time.sleep(2)
    ser.reset_input_buffer()
    
    print("Testing 'scan' command...")
    # Send the exact command from your script
    ser.write(b'scan 2000000000 3000000000 201 5\n')
    
    print("Waiting for response (5 seconds)...")
    time.sleep(5)  # Give it plenty of time
    
    # Read everything available
    response = ser.read_all().decode('ascii', errors='ignore')
    
    print("Response received:")
    print("-" * 40)
    print(response[:500])  # First 500 characters
    print("-" * 40)
    
    if len(response) > 0:
        print(f"✓ Got {len(response)} characters of response")
        # Count lines
        lines = response.strip().split('\n')
        print(f"✓ Found {len(lines)} lines")
        if len(lines) > 0:
            print("First few lines:")
            for i, line in enumerate(lines[:5]):
                print(f"  {i}: {line}")
    else:
        print("✗ No response received")