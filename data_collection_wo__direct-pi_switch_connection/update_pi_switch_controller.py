# pi_switch_controller.py
"""
Run this on your Raspberry Pi to control RF switches
Can be controlled manually OR via network from computer
"""
import RPi.GPIO as GPIO
import time
import socket
import threading
import sys

# GPIO pin configuration
# Switch 1 (TX) - Pins 17, 27
# Switch 2 (RX) - Pins 18, 22
PINS = {
    'switch1_rf1': 17,
    'switch1_rf2': 27,
    'switch2_rf1': 18,
    'switch2_rf2': 22
}

# Path configurations
PATHS = {
    1: {  # Path 1: Antenna 1 → Antenna 3
        'switch1': {'rf1': 1, 'rf2': 0},
        'switch2': {'rf1': 1, 'rf2': 0},
        'name': '1→3'
    },
    2: {  # Path 2: Antenna 1 → Antenna 4
        'switch1': {'rf1': 1, 'rf2': 0},
        'switch2': {'rf1': 0, 'rf2': 1},
        'name': '1→4'
    },
    3: {  # Path 3: Antenna 2 → Antenna 3
        'switch1': {'rf1': 0, 'rf2': 1},
        'switch2': {'rf1': 1, 'rf2': 0},
        'name': '2→3'
    },
    4: {  # Path 4: Antenna 2 → Antenna 4
        'switch1': {'rf1': 0, 'rf2': 1},
        'switch2': {'rf1': 0, 'rf2': 1},
        'name': '2→4'
    }
}

def setup_gpio():
    """Initialize GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    for pin in PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    print("✅ GPIO initialized")

def set_path(path_num):
    """Set RF switches to specified path"""
    if path_num not in PATHS:
        print(f"❌ Invalid path: {path_num}")
        return False
    
    config = PATHS[path_num]
    
    # Set switch 1 (TX)
    GPIO.output(PINS['switch1_rf1'], config['switch1']['rf1'])
    GPIO.output(PINS['switch1_rf2'], config['switch1']['rf2'])
    
    # Set switch 2 (RX)
    GPIO.output(PINS['switch2_rf1'], config['switch2']['rf1'])
    GPIO.output(PINS['switch2_rf2'], config['switch2']['rf2'])
    
    print(f"✅ Path {path_num} set: {config['name']}")
    return True

def network_server(port=8888):
    """Run network server to receive commands from computer"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(5)
    print(f"🌐 Network server running on port {port}")
    print(f"   Connect from computer using: telnet <pi_ip> {port}")
    print("   Commands: 'path 1', 'path 2', 'path 3', 'path 4', 'quit'")
    
    def handle_client(client_socket, addr):
        print(f"📡 Connected from {addr}")
        try:
            while True:
                data = client_socket.recv(1024).decode().strip()
                if not data:
                    break
                
                if data.startswith('path'):
                    try:
                        path_num = int(data.split()[1])
                        set_path(path_num)
                        client_socket.send(f"OK Path {path_num}\n".encode())
                    except:
                        client_socket.send(b"ERROR Invalid command\n")
                elif data == 'quit':
                    break
                else:
                    client_socket.send(b"ERROR Unknown command\n")
        except:
            pass
        finally:
            client_socket.close()
            print(f"📡 Disconnected from {addr}")
    
    while True:
        client, addr = server.accept()
        threading.Thread(target=handle_client, args=(client, addr), daemon=True).start()

def manual_mode():
    """Run in manual mode (keyboard input)"""
    print("\n🔧 Manual Control Mode")
    print("Enter path number (1-4) or 'q' to quit")
    
    try:
        while True:
            cmd = input("\nPath: ").strip().lower()
            if cmd == 'q':
                break
            try:
                path = int(cmd)
                if 1 <= path <= 4:
                    set_path(path)
                else:
                    print("Enter 1-4")
            except:
                print("Invalid input")
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up")

def main():
    print("="*50)
    print("PULMO AI - RF Switch Controller")
    print("="*50)
    print("\nSelect mode:")
    print("  1. Manual control (keyboard)")
    print("  2. Network server (control from computer)")
    print("  3. Demo mode (cycle through paths)")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    setup_gpio()
    
    if choice == '1':
        manual_mode()
    elif choice == '2':
        try:
            port = int(input("Port (default 8888): ") or "8888")
            network_server(port)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            GPIO.cleanup()
    elif choice == '3':
        print("\n🔄 Demo: Cycling through paths...")
        try:
            for _ in range(3):
                for path in range(1, 5):
                    set_path(path)
                    time.sleep(2)
        finally:
            GPIO.cleanup()
    else:
        print("Invalid choice")
        GPIO.cleanup()

if __name__ == "__main__":
    main()
