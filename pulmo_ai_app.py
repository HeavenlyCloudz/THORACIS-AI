#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULMO-AI Unified Application - TFLite Only Version
- Uses tflite-runtime (lightweight) for all ML inference
- Direct serial VNA communication
- Automatic RF switch control
- Saves scans as CSV files for reliable baseline storage
"""

import os

# Force OpenGL ES - must be set before any Qt imports
if 'QT_OPENGL' not in os.environ:
    os.environ['QT_OPENGL'] = 'es2'
if 'QT_QPA_EGLFS_HIDECURSOR' not in os.environ:
    os.environ['QT_QPA_EGLFS_HIDECURSOR'] = '1'

import sys
import time
import json
import threading
import serial
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle
import math
import traceback
import csv
import shutil

# Qt
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QProgressBar, QMessageBox, QTabWidget, QTextEdit, QLineEdit,
    QComboBox
)

# Hardware
import RPi.GPIO as GPIO
import sounddevice as sd

# ML
import tflite_runtime.interpreter as tflite

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path.home() / "pulmo_ai_app" / "models"
AUDIO_MODEL_PATH = MODEL_DIR / "lung_audio.tflite"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_xgboost_model.pkl"
FUSION_SCALER_PATH = MODEL_DIR / "fusion_scaler.pkl"
YAMNET_TFLITE_PATH = MODEL_DIR / "yamnet.tflite"

# Data directories
DATA_DIR = Path.home() / "pulmo_ai_app" / "scans"
BASELINE_DIR = DATA_DIR / "baseline"
PATIENT_DIR = DATA_DIR / "patient"

# VNA Serial Settings
VNA_PORT = '/dev/ttyACM0'
BAUDRATE = 115200
START_FREQ = 2000000000
STOP_FREQ = 3000000000
POINTS = 201

# GPIO pins
SWITCH1_A = 17
SWITCH1_B = 27
SWITCH2_A = 18
SWITCH2_B = 22

PATHS = {
    1: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 1, SWITCH2_B: 0, 'name': '1->3', 'desc': 'Antenna 1 to Antenna 3'},
    2: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 0, SWITCH2_B: 1, 'name': '1->4', 'desc': 'Antenna 1 to Antenna 4'},
    3: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 1, SWITCH2_B: 0, 'name': '2->3', 'desc': 'Antenna 2 to Antenna 3'},
    4: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 0, SWITCH2_B: 1, 'name': '2->4', 'desc': 'Antenna 2 to Antenna 4'},
}

# Audio settings
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
AUDIO_CLASSES = ['asthma', 'copd', 'pneumonia', 'healthy', 'bronchial']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def db_to_linear(db):
    return 10 ** (db / 10)

def linear_to_db(linear):
    linear = np.maximum(linear, 1e-12)
    return 10 * np.log10(linear)

def add_time_domain_features(X):
    n_samples, n_features = X.shape
    n_paths = 4
    freq_per_path = n_features // n_paths
    
    time_features = []
    for sample in X:
        sample_time = []
        for p in range(n_paths):
            start = p * freq_per_path
            end = (p + 1) * freq_per_path
            freq_resp = sample[start:end]
            time_resp = np.abs(np.fft.ifft(freq_resp))
            sample_time.extend([
                np.max(time_resp), np.argmax(time_resp),
                np.mean(time_resp), np.std(time_resp),
                np.percentile(time_resp, 90), np.percentile(time_resp, 10),
                np.sum(time_resp), np.max(time_resp) - np.min(time_resp),
                np.sum(time_resp ** 2)
            ])
        time_features.append(sample_time)
    time_features = np.array(time_features)
    return np.concatenate([X, time_features], axis=1)

# =============================================================================
# VNA DIRECT SERIAL CONTROLLER
# =============================================================================
class VNADirectController:
    def __init__(self, port=VNA_PORT, baudrate=BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.frequencies = None
        self.connect()
    
    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=3)
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            print(f"VNA connected on {self.port}")
            return True
        except Exception as e:
            print(f"VNA connection failed on {self.port}: {e}")
            return False
    
    def capture_s21(self, progress_callback=None):
        """Capture S21 data for current RF path"""
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return None
        
        try:
            # Clear buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Send scan command
            cmd = f"scan {START_FREQ} {STOP_FREQ} {POINTS} 5\r\n"
            self.serial_conn.write(cmd.encode())
            print(f"Sent command: {cmd.strip()}")
            
            # Wait for data
            time.sleep(2.0)
            
            data_points = []
            lines_collected = 0
            timeout_start = time.time()
            max_timeout = 20  # seconds
            
            while lines_collected < POINTS and (time.time() - timeout_start) < max_timeout:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
                    
                    # Skip non-data lines
                    if not line or line.startswith('ch>') or line.startswith('scan') or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            freq_hz = float(parts[0])
                            s21_real = float(parts[1])
                            s21_imag = float(parts[2])
                            
                            magnitude = math.sqrt(s21_real**2 + s21_imag**2)
                            magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -120
                            
                            data_points.append(magnitude_db)
                            lines_collected += 1
                            
                            if progress_callback and lines_collected % 20 == 0:
                                progress_callback(lines_collected, POINTS)
                        except ValueError:
                            continue
                else:
                    time.sleep(0.05)
            
            print(f"Collected {len(data_points)}/{POINTS} points")
            
            if len(data_points) == POINTS:
                if self.frequencies is None:
                    self.frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, POINTS)
                return np.array(data_points)
            else:
                print(f"Only {len(data_points)}/{POINTS} points captured")
                return None
                
        except Exception as e:
            print(f"VNA capture error: {e}")
            traceback.print_exc()
            return None
    
    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("VNA disconnected")

# =============================================================================
# RF SWITCH CONTROLLER
# =============================================================================
class RFSwitchController:
    def __init__(self):
        self._setup_gpio()
    
    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in [SWITCH1_A, SWITCH1_B, SWITCH2_A, SWITCH2_B]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)
        print("GPIO initialized for RF switches")
    
    def set_path(self, path_num):
        if path_num not in PATHS:
            raise ValueError(f"Invalid path: {path_num}")
        states = PATHS[path_num]
        for pin, state in states.items():
            if pin in [SWITCH1_A, SWITCH1_B, SWITCH2_A, SWITCH2_B]:
                GPIO.output(pin, state)
        time.sleep(0.1)
        print(f"Path {path_num} set")
    
    def cleanup(self):
        GPIO.cleanup()
        print("GPIO cleaned up")

# =============================================================================
# CSV DATA MANAGER
# =============================================================================
class CSVDataManager:
    def __init__(self):
        # Create directories if they don't exist
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        PATIENT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Data directories created: {DATA_DIR}")
    
    def save_scan(self, data, path_num, directory, frequencies=None):
        """Save a single path scan to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"path{path_num}_{timestamp}.csv"
        filepath = directory / filename
        
        # Prepare data rows
        if frequencies is None:
            frequencies = np.linspace(START_FREQ/1e9, STOP_FREQ/1e9, len(data))
        
        rows = []
        for i, (freq, s21) in enumerate(zip(frequencies, data)):
            rows.append([freq, s21])
        
        # Save to CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frequency_GHz', 'S21_dB'])
            writer.writerows(rows)
        
        return filepath
    
    def load_latest_baseline(self):
        """Load the most recent baseline scan for each path"""
        if not BASELINE_DIR.exists():
            return None
        
        baseline_data = {}
        for path_num in [1, 2, 3, 4]:
            # Find latest CSV for this path
            files = list(BASELINE_DIR.glob(f"path{path_num}_*.csv"))
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest)
                baseline_data[path_num] = df['S21_dB'].values
        
        return baseline_data if baseline_data else None
    
    def load_latest_patient(self):
        """Load the most recent patient scan for each path"""
        if not PATIENT_DIR.exists():
            return None
        
        patient_data = {}
        for path_num in [1, 2, 3, 4]:
            files = list(PATIENT_DIR.glob(f"path{path_num}_*.csv"))
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest)
                patient_data[path_num] = df['S21_dB'].values
        
        return patient_data if patient_data else None
    
    def has_baseline(self):
        """Check if baseline data exists"""
        for path_num in [1, 2, 3, 4]:
            files = list(BASELINE_DIR.glob(f"path{path_num}_*.csv"))
            if files:
                return True
        return False
    
    def clear_baseline(self):
        """Clear all baseline files"""
        if BASELINE_DIR.exists():
            for f in BASELINE_DIR.glob("*.csv"):
                f.unlink()
            print("Baseline cleared")
    
    def clear_patient(self):
        """Clear all patient files"""
        if PATIENT_DIR.exists():
            for f in PATIENT_DIR.glob("*.csv"):
                f.unlink()
            print("Patient scans cleared")
    
    def clear_all(self):
        """Clear all data"""
        self.clear_baseline()
        self.clear_patient()
        print("All data cleared")

# =============================================================================
# MICROWAVE SCANNER WITH CSV STORAGE
# =============================================================================
class MicrowaveScanner:
    def __init__(self, vna_controller):
        self.vna = vna_controller
        self.switch = RFSwitchController()
        self.csv_manager = CSVDataManager()
        self.frequencies = None
    
    def scan_and_save_all_paths(self, save_dir, progress_callback=None):
        """Scan all 4 paths and save to CSV files"""
        data = {}
        total_paths = len(PATHS)
        
        for idx, path_num in enumerate(PATHS.keys(), 1):
            if progress_callback:
                progress_callback(f"Setting Path {path_num}", idx / total_paths)
            
            self.switch.set_path(path_num)
            time.sleep(0.2)
            
            if progress_callback:
                progress_callback(f"Capturing Path {path_num}", idx / total_paths)
            
            s21_data = self.vna.capture_s21()
            if s21_data is None:
                raise RuntimeError(f"Failed to capture path {path_num}")
            
            data[path_num] = s21_data
            
            if self.frequencies is None:
                self.frequencies = self.vna.frequencies
            
            # Save to CSV
            self.csv_manager.save_scan(s21_data, path_num, save_dir, self.frequencies)
            
            if progress_callback:
                progress_callback(f"Path {path_num} complete", idx / total_paths)
        
        return data
    
    def load_baseline(self):
        """Load baseline data from CSV files"""
        return self.csv_manager.load_latest_baseline()
    
    def load_patient(self):
        """Load patient data from CSV files"""
        return self.csv_manager.load_latest_patient()
    
    def has_baseline(self):
        """Check if baseline exists"""
        return self.csv_manager.has_baseline()
    
    def clear_data(self):
        """Clear all saved data"""
        self.csv_manager.clear_all()
    
    def subtract_baseline(self, patient_data, baseline_data):
        """Apply background subtraction"""
        if baseline_data is None:
            raise ValueError("No baseline data available")
        
        corrected = {}
        for path_num in patient_data.keys():
            if path_num in baseline_data:
                air_linear = db_to_linear(baseline_data[path_num])
                patient_linear = db_to_linear(patient_data[path_num])
                corrected_linear = patient_linear - air_linear
                corrected[path_num] = linear_to_db(corrected_linear)
            else:
                corrected[path_num] = patient_data[path_num]
        
        return corrected
    
    def extract_features(self, s21_data):
        """Convert S21 dict to feature vector"""
        raw = np.array([s21_data[p] for p in [1,2,3,4]]).reshape(1, -1)
        return add_time_domain_features(raw)[0]
    
    def cleanup(self):
        self.switch.cleanup()

# =============================================================================
# Audio Processor (Fixed with device selection)
# =============================================================================
class AcousticProcessor(QThread):
    result_ready = Signal(np.ndarray)
    waveform_ready = Signal(np.ndarray)
    finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self, record_seconds=3, device_id=None):
        super().__init__()
        self.record_seconds = record_seconds
        self.device_id = device_id
        self.classifier = None
        self.yamnet = None
        self._load_models()
    
    def _load_models(self):
        """Load audio models with error handling"""
        try:
            # Check if audio model exists
            if not AUDIO_MODEL_PATH.exists():
                self.error_occurred.emit(f"Audio model not found: {AUDIO_MODEL_PATH}")
                print(f"ERROR: Audio model not found at {AUDIO_MODEL_PATH}")
                return
            
            # Load classifier
            self.classifier = tflite.Interpreter(str(AUDIO_MODEL_PATH))
            self.classifier.allocate_tensors()
            self.classifier_input = self.classifier.get_input_details()[0]
            self.classifier_output = self.classifier.get_output_details()[0]
            print(f"Audio classifier loaded from {AUDIO_MODEL_PATH}")
            print(f"Input shape: {self.classifier_input['shape']}, dtype: {self.classifier_input['dtype']}")
            
            # Load YAMNet if available
            if YAMNET_TFLITE_PATH.exists():
                self.yamnet = tflite.Interpreter(str(YAMNET_TFLITE_PATH))
                self.yamnet.allocate_tensors()
                self.yamnet_input = self.yamnet.get_input_details()[0]
                self.yamnet_output = self.yamnet.get_output_details()
                print("YAMNet loaded")
            else:
                print(f"YAMNet not found at {YAMNET_TFLITE_PATH}, using simple features")
                
        except Exception as e:
            self.error_occurred.emit(f"Failed to load audio models: {e}")
            print(f"Audio model loading error: {e}")
            traceback.print_exc()
    
    def _extract_yamnet_features(self, audio):
        """Extract features using YAMNet"""
        try:
            # Reshape for YAMNet
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
            
            # Set input
            self.yamnet.set_tensor(self.yamnet_input['index'], audio.astype(np.float32))
            
            # Run inference
            self.yamnet.invoke()
            
            # Get embeddings (usually the second output)
            embeddings = self.yamnet.get_tensor(self.yamnet_output[1]['index'])
            return np.mean(embeddings, axis=0)
        except Exception as e:
            print(f"YAMNet feature extraction error: {e}")
            return None
    
    def _extract_simple_features(self, audio):
        """Simple fallback features"""
        # Simple spectral features
        fft = np.abs(np.fft.rfft(audio, n=2048))[:512]
        fft = fft / (np.max(fft) + 1e-6)
        
        # Time-domain features
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)
        energy = np.sum(audio ** 2) / len(audio)
        
        # Combine
        features = np.concatenate([fft[:200], [zcr, energy]])
        
        # Pad to expected input size (assuming 1024)
        if len(features) < 1024:
            features = np.pad(features, (0, 1024 - len(features)))
        else:
            features = features[:1024]
        
        return features.astype(np.float32)
    
    def run(self):
        """Capture audio and run inference"""
        try:
            print(f"Starting audio capture for {self.record_seconds} seconds...")
            if self.device_id is not None:
                print(f"Using audio device: {self.device_id}")
            
            # Record audio with proper duration using specified device
            sample_count = int(self.record_seconds * SAMPLE_RATE)
            print(f"Recording {sample_count} samples at {SAMPLE_RATE} Hz")
            
            recording = sd.rec(sample_count,
                               samplerate=SAMPLE_RATE, 
                               channels=1, 
                               dtype='float32',
                               device=self.device_id,
                               blocking=True)
            
            # Wait for recording to complete
            sd.wait()
            
            audio = recording.flatten()
            print(f"Recorded {len(audio)} samples, duration: {len(audio)/SAMPLE_RATE:.2f} seconds")
            
            # Check if we got enough audio
            if len(audio) < sample_count * 0.8:
                print(f"WARNING: Only got {len(audio)} samples, expected {sample_count}")
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            print(f"Audio normalized, max amplitude: {np.max(np.abs(audio)):.3f}")
            
            # Extract features
            if self.yamnet is not None:
                features = self._extract_yamnet_features(audio)
                if features is not None:
                    input_data = features.reshape(1, -1).astype(np.float32)
                else:
                    input_data = self._extract_simple_features(audio).reshape(1, -1)
            else:
                input_data = self._extract_simple_features(audio).reshape(1, -1)
            
            print(f"Feature shape: {input_data.shape}")
            
            # Run classifier
            if self.classifier is not None:
                # Ensure input has correct shape
                expected_shape = self.classifier_input['shape']
                print(f"Expected input shape: {expected_shape}")
                
                self.classifier.set_tensor(self.classifier_input['index'], input_data)
                self.classifier.invoke()
                probs = self.classifier.get_tensor(self.classifier_output['index'])[0]
                print(f"Audio probabilities: {probs}")
                self.result_ready.emit(probs.astype(np.float32))
            else:
                # Return zeros if classifier not loaded
                self.result_ready.emit(np.zeros(5, dtype=np.float32))
                self.error_occurred.emit("Audio classifier not loaded")
            
            # Generate waveform for display
            if len(audio) > 800:
                step = max(1, len(audio) // 800)
                down = audio[::step][:800]
                self.waveform_ready.emit(down)
                
        except Exception as e:
            print(f"Audio processing error: {e}")
            traceback.print_exc()
            self.error_occurred.emit(f"Audio error: {e}")
            self.result_ready.emit(np.zeros(5, dtype=np.float32))
        finally:
            self.finished.emit()

# =============================================================================
# Fusion Classifier
# =============================================================================
class FusionClassifier:
    def __init__(self):
        if not FUSION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Fusion model not found: {FUSION_MODEL_PATH}")
        if not FUSION_SCALER_PATH.exists():
            raise FileNotFoundError(f"Fusion scaler not found: {FUSION_SCALER_PATH}")
        
        with open(FUSION_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(FUSION_SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Fusion model loaded")
    
    def predict(self, mw_features, audio_probs):
        fusion_vec = np.concatenate([mw_features, audio_probs]).reshape(1, -1)
        scaled = self.scaler.transform(fusion_vec)
        pred = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        return pred, np.max(proba)

# =============================================================================
# Main GUI Application
# =============================================================================
class PulmoAIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PULMO-AI: Lung Screening System")
        
        # Initialize VNA and scanner
        self.vna = VNADirectController()
        self.scanner = MicrowaveScanner(self.vna)
        
        # Audio device selection
        self.audio_device_id = None
        self._setup_audio_devices()
        
        try:
            self.fusion = FusionClassifier()
        except Exception as e:
            print(f"Fusion not loaded: {e}")
            self.fusion = None
        
        self.current_mw_features = None
        self.current_audio_probs = None
        self.current_s21_data = None
        self.baseline_data = None
        self.patient_data = None
        
        self._setup_ui()
        self.showFullScreen()
    
    def _setup_audio_devices(self):
        """Detect and select the correct USB audio input device"""
        try:
            devices = sd.query_devices()
            print("Available audio devices:")
            for i, device in enumerate(devices):
                print(f"  {i}: {device['name']} - {device['max_input_channels']} in, {device['max_output_channels']} out")
            
            # Find first device with input channels that contains "USB Audio" in name
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and 'USB Audio' in device['name']:
                    self.audio_device_id = i
                    print(f"Selected audio input device: {i} - {device['name']}")
                    break
            
            if self.audio_device_id is None:
                print("WARNING: No USB audio input device found. Using default device.")
        except Exception as e:
            print(f"Error detecting audio devices: {e}")
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("PULMO-AI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #0277bd;
            background-color: #e1f5fe;
            padding: 20px;
            border-radius: 20px;
            margin-bottom: 10px;
        """)
        layout.addWidget(title)
        
        # Status bar
        vna_status = "Connected" if self.vna.serial_conn else "Disconnected"
        audio_status = f"Audio: {'USB' if self.audio_device_id is not None else 'Default'}"
        self.status_bar = QLabel(f"VNA: {vna_status} on {VNA_PORT} | {audio_status}")
        self.status_bar.setStyleSheet("font-size: 12px; color: gray; padding: 5px;")
        layout.addWidget(self.status_bar)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #4fc3f7;
                border-radius: 10px;
                background: white;
            }
            QTabBar::tab {
                font-size: 18px;
                font-weight: bold;
                padding: 12px 24px;
                background: #e1f5fe;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 5px;
            }
            QTabBar::tab:selected {
                background: #4fc3f7;
                color: white;
            }
        """)
        layout.addWidget(self.tabs)
        
        self._add_microwave_tab()
        self._add_audio_tab()
        self._add_fusion_tab()
        
        # Exit button
        exit_btn = QPushButton("EXIT")
        exit_btn.setMinimumHeight(50)
        exit_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background: #ef5350;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px;
                margin-top: 10px;
            }
            QPushButton:hover { background: #ff6659; }
        """)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)
    
    def _add_microwave_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        self.mw_status = QLabel("Ready")
        self.mw_status.setStyleSheet("font-size: 14px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.mw_status)
        
        self.baseline_btn = QPushButton("1. RECORD BASELINE (AIR)")
        self.baseline_btn.setMinimumHeight(60)
        self.baseline_btn.setStyleSheet(self._button_style("#81d4fa"))
        self.baseline_btn.clicked.connect(self._record_baseline)
        layout.addWidget(self.baseline_btn)
        
        self.scan_btn = QPushButton("2. SCAN PATIENT")
        self.scan_btn.setMinimumHeight(80)
        self.scan_btn.setStyleSheet(self._button_style("#4fc3f7"))
        self.scan_btn.clicked.connect(self._run_microwave_scan)
        layout.addWidget(self.scan_btn)
        
        self.mw_progress = QProgressBar()
        self.mw_progress.setVisible(False)
        layout.addWidget(self.mw_progress)
        
        self.mw_result = QTextEdit()
        self.mw_result.setReadOnly(True)
        self.mw_result.setMinimumHeight(150)
        layout.addWidget(self.mw_result)
        
        # Clear Data button
        clear_btn = QPushButton("CLEAR ALL DATA")
        clear_btn.setMinimumHeight(50)
        clear_btn.setStyleSheet(self._button_style("#ff9800"))
        clear_btn.clicked.connect(self._clear_all_data)
        layout.addWidget(clear_btn)
        
        self.reconstruct_btn = QPushButton("RECONSTRUCT IMAGE")
        self.reconstruct_btn.setEnabled(False)
        self.reconstruct_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.reconstruct_btn.clicked.connect(self._show_reconstruction)
        layout.addWidget(self.reconstruct_btn)
        
        self.tabs.addTab(tab, "Microwave Scan")
    
    def _add_audio_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        self.audio_status = QLabel("Ready")
        self.audio_status.setStyleSheet("font-size: 14px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.audio_status)
        
        self.waveform_label = QLabel()
        self.waveform_label.setMinimumHeight(150)
        self.waveform_label.setStyleSheet("background-color: black; border-radius: 8px;")
        layout.addWidget(self.waveform_label)
        
        self.audio_btn = QPushButton("ANALYZE LUNG SOUNDS")
        self.audio_btn.setMinimumHeight(80)
        self.audio_btn.setStyleSheet(self._button_style("#66bb6a"))
        self.audio_btn.clicked.connect(self._run_acoustic_analysis)
        layout.addWidget(self.audio_btn)
        
        self.audio_result = QTextEdit()
        self.audio_result.setReadOnly(True)
        self.audio_result.setMinimumHeight(150)
        layout.addWidget(self.audio_result)
        
        self.tabs.addTab(tab, "Acoustic Analysis")
    
    def _add_fusion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        self.fusion_status = QLabel("Perform both scans for combined diagnosis")
        self.fusion_status.setStyleSheet("font-size: 14px; padding: 8px; background: #fff3e0; border-radius: 8px;")
        layout.addWidget(self.fusion_status)
        
        self.fusion_mw_btn = QPushButton("1. SCAN MICROWAVE")
        self.fusion_mw_btn.setMinimumHeight(60)
        self.fusion_mw_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_mw_btn.clicked.connect(self._fusion_microwave)
        layout.addWidget(self.fusion_mw_btn)
        
        self.fusion_audio_btn = QPushButton("2. ANALYZE ACOUSTIC")
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_audio_btn.setMinimumHeight(60)
        self.fusion_audio_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_audio_btn.clicked.connect(self._fusion_acoustic)
        layout.addWidget(self.fusion_audio_btn)
        
        self.fusion_combine_btn = QPushButton("3. RUN FUSION DIAGNOSIS")
        self.fusion_combine_btn.setEnabled(False)
        self.fusion_combine_btn.setMinimumHeight(80)
        self.fusion_combine_btn.setStyleSheet(self._button_style("#4fc3f7"))
        self.fusion_combine_btn.clicked.connect(self._run_fusion)
        layout.addWidget(self.fusion_combine_btn)
        
        self.fusion_result = QTextEdit()
        self.fusion_result.setReadOnly(True)
        self.fusion_result.setMinimumHeight(150)
        layout.addWidget(self.fusion_result)
        
        self.tabs.addTab(tab, "Full Fusion")
    
    def _button_style(self, color):
        return f"""
            QPushButton {{
                font-size: 18px;
                font-weight: bold;
                background: {color};
                color: white;
                border: none;
                border-radius: 15px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background: {color}cc;
            }}
            QPushButton:disabled {{
                background: #cccccc;
                color: #888;
            }}
        """
    
    def _update_mw_progress(self, msg, frac):
        self.mw_status.setText(msg)
        self.mw_progress.setValue(int(frac * 100))
        QApplication.processEvents()
    
    def _clear_all_data(self):
        """Clear all saved CSV files"""
        reply = QMessageBox.question(
            self, "Clear Data", 
            "This will delete all saved baseline and patient scans. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.scanner.clear_data()
            self.current_mw_features = None
            self.current_audio_probs = None
            self.current_s21_data = None
            self.baseline_data = None
            self.patient_data = None
            self.reconstruct_btn.setEnabled(False)
            self.mw_result.setText("All data cleared. You can now record a new baseline.")
            self.mw_status.setText("Ready")
            QMessageBox.information(self, "Data Cleared", "All saved scans have been deleted.")
    
    def _record_baseline(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected. Please check USB connection.")
            return
        
        self.baseline_btn.setEnabled(False)
        self.mw_status.setText("Recording baseline (air)...")
        self.mw_progress.setVisible(True)
        self.mw_progress.setValue(0)
        self.mw_result.setText("Starting baseline scan...")
        
        def worker():
            try:
                # Scan all 4 paths and save to baseline directory
                data = self.scanner.scan_and_save_all_paths(
                    BASELINE_DIR,
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.baseline_data = data
                
                # Calculate average S21 for feedback
                avg_s21 = np.mean([np.mean(data[p]) for p in [1,2,3,4]])
                result_text = f"Baseline recorded successfully!\n\n"
                result_text += f"Average S21: {avg_s21:.1f} dB\n\n"
                result_text += f"Path values:\n"
                for p in [1,2,3,4]:
                    result_text += f"  Path {p}: {np.mean(data[p]):.1f} dB\n"
                result_text += f"\nFiles saved to: {BASELINE_DIR}\n"
                result_text += f"\nNow place patient and click SCAN PATIENT."
                
                self.mw_result.setText(result_text)
                self.mw_status.setText("Baseline complete")
                
            except Exception as e:
                error_msg = f"Error during baseline: {e}\n\nCheck VNA connection and try again."
                self.mw_result.setText(error_msg)
                self.mw_status.setText("Baseline failed")
                print(f"Baseline error: {e}")
                traceback.print_exc()
            finally:
                self.baseline_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _run_microwave_scan(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected. Please check USB connection.")
            return
        
        if not self.scanner.has_baseline():
            QMessageBox.warning(self, "Missing Baseline", "Please record baseline (air) first!")
            return
        
        self.scan_btn.setEnabled(False)
        self.mw_progress.setVisible(True)
        self.mw_progress.setValue(0)
        self.mw_result.setText("Scanning patient...")
        
        def worker():
            try:
                # Scan all 4 paths and save to patient directory
                data = self.scanner.scan_and_save_all_paths(
                    PATIENT_DIR,
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.patient_data = data
                
                # Load baseline data from CSV
                baseline_data = self.scanner.load_baseline()
                
                # Apply background subtraction
                corrected = self.scanner.subtract_baseline(data, baseline_data)
                features = self.scanner.extract_features(corrected)
                self.current_mw_features = features
                self.current_s21_data = data
                
                # Calculate results
                result_text = f"Scan complete!\n\n"
                result_text += f"Raw S21 values:\n"
                for p in [1,2,3,4]:
                    result_text += f"  Path {p}: {np.mean(data[p]):.1f} dB\n"
                result_text += f"\nBackground subtracted values:\n"
                for p in [1,2,3,4]:
                    result_text += f"  Path {p}: {np.mean(corrected[p]):.1f} dB\n"
                result_text += f"\nFiles saved to: {PATIENT_DIR}\n"
                result_text += f"\nReady for fusion."
                
                self.mw_result.setText(result_text)
                self.reconstruct_btn.setEnabled(True)
                self.mw_status.setText("Scan complete")
            except Exception as e:
                self.mw_result.setText(f"Error during scan: {e}")
                self.mw_status.setText("Scan failed")
                print(f"Scan error: {e}")
                traceback.print_exc()
            finally:
                self.scan_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _show_reconstruction(self):
        if self.current_s21_data is None:
            QMessageBox.information(self, "No Data", "Perform a scan first.")
            return
        
        s21_values = [np.mean(self.current_s21_data[p]) for p in [1,2,3,4]]
        msg = "Path-wise S21 averages:\n\n"
        msg += f"Path 1 (1->3): {s21_values[0]:.1f} dB\n"
        msg += f"Path 2 (1->4): {s21_values[1]:.1f} dB\n"
        msg += f"Path 3 (2->3): {s21_values[2]:.1f} dB\n"
        msg += f"Path 4 (2->4): {s21_values[3]:.1f} dB\n\n"
        msg += f"Spatial asymmetry: {abs(s21_values[0] - s21_values[3]):.1f} dB\n"
        
        max_diff = max(s21_values) - min(s21_values)
        if max_diff > 5:
            msg += f"\nTumor indicator: POSITIVE (asymmetry > 5 dB)"
        else:
            msg += f"\nTumor indicator: Negative (asymmetry {max_diff:.1f} dB)"
        
        QMessageBox.information(self, "Microwave Analysis", msg)
    
    def _run_acoustic_analysis(self):
        self.audio_btn.setEnabled(False)
        self.audio_status.setText("Recording (3 seconds)...")
        self.audio_result.setText("Processing...")
        self.waveform_label.setText("")  # Clear waveform
        
        # Use the selected audio device
        self.audio_thread = AcousticProcessor(RECORD_SECONDS, self.audio_device_id)
        self.audio_thread.result_ready.connect(self._on_audio_result)
        self.audio_thread.waveform_ready.connect(self._draw_waveform)
        self.audio_thread.error_occurred.connect(self._on_audio_error)
        self.audio_thread.finished.connect(self._on_audio_finished)
        self.audio_thread.start()
    
    def _on_audio_result(self, probs):
        self.current_audio_probs = probs
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        class_name = AUDIO_CLASSES[class_idx]
        
        result_text = f"PREDICTION: {class_name.upper()}\n"
        result_text += f"Confidence: {confidence:.1%}\n\n"
        result_text += "Detailed Probabilities:\n"
        for c, p in zip(AUDIO_CLASSES, probs):
            result_text += f"   {c}: {p:.1%}\n"
        
        self.audio_result.setText(result_text)
        self.audio_status.setText(f"Result: {class_name} ({confidence:.1%})")
        
        # Update fusion tab if microwave data exists
        if self.current_mw_features is not None and self.fusion is not None:
            self.fusion_combine_btn.setEnabled(True)
    
    def _on_audio_error(self, error_msg):
        self.audio_result.setText(f"Audio Error: {error_msg}\n\nCheck microphone and try again.")
        self.audio_status.setText("Audio error")
        print(f"Audio error: {error_msg}")
    
    def _on_audio_finished(self):
        self.audio_btn.setEnabled(True)
    
    def _draw_waveform(self, data):
        if data is None or len(data) == 0:
            return
        w = self.waveform_label.width()
        h = self.waveform_label.height()
        if w < 10 or h < 10:
            return
        
        pixmap = QtGui.QPixmap(w, h)
        pixmap.fill(Qt.black)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.cyan, 2))
        
        mid = h // 2
        data = (data - np.mean(data)) / (np.std(data) + 1e-6)
        data = np.clip(data, -1, 1)
        step = w / len(data)
        
        for i in range(1, len(data)):
            x1 = int((i-1) * step)
            x2 = int(i * step)
            y1 = int(mid + data[i-1] * mid)
            y2 = int(mid + data[i] * mid)
            painter.drawLine(x1, y1, x2, y2)
        
        painter.end()
        self.waveform_label.setPixmap(pixmap)
    
    def _fusion_microwave(self):
        if not self.vna.serial_conn:
            QMessageBox.warning(self, "VNA Error", "VNA not connected. Please check USB connection.")
            return
        
        self.fusion_mw_btn.setEnabled(False)
        self.fusion_status.setText("Scanning microwave...")
        
        def worker():
            try:
                if not self.scanner.has_baseline():
                    self.fusion_status.setText("Need baseline first! Go to Microwave tab.")
                    self.fusion_mw_btn.setEnabled(True)
                    return
                
                # Scan patient
                data = self.scanner.scan_and_save_all_paths(PATIENT_DIR)
                self.patient_data = data
                
                # Load baseline and apply subtraction
                baseline_data = self.scanner.load_baseline()
                corrected = self.scanner.subtract_baseline(data, baseline_data)
                features = self.scanner.extract_features(corrected)
                self.current_mw_features = features
                self.fusion_status.setText("Microwave complete. Now perform acoustic.")
                self.fusion_audio_btn.setEnabled(True)
            except Exception as e:
                self.fusion_status.setText(f"Error: {e}")
                print(f"Fusion microwave error: {e}")
            finally:
                self.fusion_mw_btn.setEnabled(True)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _fusion_acoustic(self):
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_status.setText("Recording acoustic...")
        self.audio_thread = AcousticProcessor(RECORD_SECONDS, self.audio_device_id)
        self.audio_thread.result_ready.connect(self._on_fusion_audio_result)
        self.audio_thread.error_occurred.connect(self._on_fusion_audio_error)
        self.audio_thread.finished.connect(self._on_fusion_audio_finished)
        self.audio_thread.start()
    
    def _on_fusion_audio_result(self, probs):
        self.current_audio_probs = probs
        self.fusion_status.setText("Acoustic complete. Ready for fusion.")
        self.fusion_combine_btn.setEnabled(True)
    
    def _on_fusion_audio_error(self, error_msg):
        self.fusion_status.setText(f"Acoustic error: {error_msg}")
        self.fusion_audio_btn.setEnabled(True)
    
    def _on_fusion_audio_finished(self):
        self.fusion_audio_btn.setEnabled(True)
    
    def _run_fusion(self):
        if self.current_mw_features is None or self.current_audio_probs is None:
            QMessageBox.warning(self, "Missing Data", "Perform both scans first!")
            return
        
        if self.fusion is None:
            self.fusion_result.setText("Fusion model not loaded!")
            return
        
        try:
            pred, conf = self.fusion.predict(self.current_mw_features, self.current_audio_probs)
            class_names = ['baseline', 'healthy', 'tumor']
            result = class_names[pred] if pred < 3 else "unknown"
            audio_class = AUDIO_CLASSES[np.argmax(self.current_audio_probs)]
            
            result_text = f"FUSION DIAGNOSIS: {result.upper()}\n\n"
            result_text += f"Confidence: {conf:.1%}\n\n"
            result_text += "Interpretation:\n"
            result_text += f"   Structural anomaly: {['None', 'Possible', 'Definite'][pred]}\n"
            result_text += f"   Functional pattern: {audio_class}\n\n"
            result_text += "Clinical Note:\n"
            result_text += f"   {'Normal findings' if result == 'baseline' else 'Further evaluation recommended'}"
            
            self.fusion_result.setText(result_text)
            self.fusion_status.setText(f"Diagnosis: {result}")
        except Exception as e:
            self.fusion_result.setText(f"Fusion error: {e}")
            print(f"Fusion error: {e}")
    
    def closeEvent(self, event):
        print("\nShutting down PULMO-AI...")
        self.scanner.cleanup()
        self.vna.close()
        event.accept()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import os

    platform = os.environ.get('QT_QPA_PLATFORM', '')
    if platform == 'eglfs':
        QApplication.setAttribute(Qt.AA_UseOpenGLES, True)

    app = QApplication(sys.argv)
    window = PulmoAIMainWindow()
    window.showFullScreen()
    sys.exit(app.exec())
