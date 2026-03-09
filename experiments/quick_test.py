# Save as quick_test.py and run it
import serial, time
ser = serial.Serial('COM3', 115200, timeout=2)
time.sleep(2)
ser.write(b'scan 2000000000 3000000000 11 5\n')  # Quick 11-point scan
time.sleep(1)
data = ser.read_all().decode()
print("First 3 points:", data[:200])
ser.close()