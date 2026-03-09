import serial
import time

PORT = 'COM4'
BAUDRATE = 115200

print("=== NANOVNA CONNECTION TEST ===")
print(f"Attempting to connect to {PORT}...")

try:
    # Open the serial port
    with serial.Serial(PORT, BAUDRATE, timeout=2) as ser:
        time.sleep(2)  # Crucial settling time
        ser.reset_input_buffer()  # Clear any junk data
        
        print("Connected! Querying device info...")
        
        # Send the 'info' command
        ser.write(b'info\n')
        time.sleep(0.5)  # Wait for response
        
        # Read all available response
        response = ser.read_all().decode('ascii', errors='ignore')
        
        print("\n--- VNA RESPONSE ---")
        print(response.strip())
        print("--- END RESPONSE ---")
        
        # Check if we got a valid response
        if 'NanoVNA' in response or 'V3' in response or 'sysjoint' in response.lower():
            print("\n✅ SUCCESS! VNA is communicating properly.")
            print("You are ready for the phantom experiments.")
        else:
            print("\n⚠️  Got a response, but it doesn't look like standard VNA info.")
            print("The connection is working, but let's verify with another command.")
            
except serial.SerialException as e:
    print(f"\n❌ SERIAL CONNECTION FAILED: {e}")
    print("Check: 1) VNA is ON, 2) USB cable is secure, 3) No other program is using COM3 (like NanoVNA-Saver).")
except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")

input("\nPress Enter to exit...")
