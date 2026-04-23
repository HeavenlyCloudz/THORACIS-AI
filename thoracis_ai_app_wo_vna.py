#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THORACIS‑AI Unified Application – TFLite Only Version
- Uses tflite-runtime (lightweight) for all ML inference
- No TensorFlow dependency
- Network VNA communication
"""

import os

# Force OpenGL ES – must be set before any Qt imports
if 'QT_OPENGL' not in os.environ:
    os.environ['QT_OPENGL'] = 'es2'
# Do NOT set QT_QPA_PLATFORM here – let the shell decide
if 'QT_QPA_EGLFS_HIDECURSOR' not in os.environ:
    os.environ['QT_QPA_EGLFS_HIDECURSOR'] = '1'

import sys
import time
import json
import threading
import socket
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle

# Qt (now that environment is set, import safely)
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QProgressBar, QMessageBox, QTabWidget, QTextEdit, QLineEdit
)

# Hardware / communication
import RPi.GPIO as GPIO
import sounddevice as sd

# ML - TFLite only
import tflite_runtime.interpreter as tflite

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path.home() / "pulmo_ai_app" / "models"
AUDIO_MODEL_PATH = MODEL_DIR / "lung_audio.tflite"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_xgboost_model.pkl"
FUSION_SCALER_PATH = MODEL_DIR / "fusion_scaler.pkl"
YAMNET_TFLITE_PATH = MODEL_DIR / "yamnet.tflite"

# Network VNA settings
VNA_SERVER_IP = "192.168.1.77" 
VNA_SERVER_PORT = 9999

# GPIO pins (RF switches)
SWITCH1_A = 17
SWITCH1_B = 27
SWITCH2_A = 18
SWITCH2_B = 22

PATHS = {
    1: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 1, SWITCH2_B: 0},
    2: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 0, SWITCH2_B: 1},
    3: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 1, SWITCH2_B: 0},
    4: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 0, SWITCH2_B: 1},
}

# Audio settings
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
AUDIO_CLASSES = ['asthma', 'copd', 'pneumonia', 'healthy', 'Bronchial']

# =============================================================================
# TIME-DOMAIN FEATURES
# =============================================================================
def add_time_domain_features(X):
    """Add IFFT-based time-domain features (matches training)"""
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
# YAMNet TFLite Wrapper (Pure TFLite)
# =============================================================================
class YAMNetTFLite:
    """Wrapper for YAMNet TFLite model"""
    
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(str(model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # YAMNet outputs: (scores, embeddings, spectrogram)
        # We need the embeddings (output index 1 typically)
        print(f"YAMNet loaded from {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Outputs: {len(self.output_details)} tensors")
    
    def extract_embeddings(self, audio):
        """
        Extract 1024-dim embeddings from audio
        audio: numpy array of shape (16000 * seconds,) float32
        """
        # Ensure correct shape (1, time)
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)
        
        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], audio)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get embeddings (usually output 1)
        # Try to find which output is embeddings (1024-dim)
        for i, output in enumerate(self.output_details):
            shape = output['shape']
            if len(shape) == 2 and shape[1] == 1024:
                embeddings = self.interpreter.get_tensor(output['index'])
                # Average over time dimension
                return np.mean(embeddings, axis=0)
        
        # Fallback: use first output
        embeddings = self.interpreter.get_tensor(self.output_details[0]['index'])
        if len(embeddings.shape) == 3:
            # Shape (1, time, 1024)
            return np.mean(embeddings[0], axis=0)
        else:
            return embeddings[0]

# =============================================================================
# Audio Processor (TFLite Only)
# =============================================================================
class AcousticProcessor(QThread):
    result_ready = Signal(np.ndarray)   # 5-class probabilities
    waveform_ready = Signal(np.ndarray)
    finished = Signal()
    
    def __init__(self, record_seconds=3):
        super().__init__()
        self.record_seconds = record_seconds
        self._load_models()
    
    def _load_models(self):
        """Load YAMNet and audio classifier using TFLite only"""
        # Load YAMNet TFLite
        if not YAMNET_TFLITE_PATH.exists():
            print(f"⚠️ YAMNet TFLite not found at {YAMNET_TFLITE_PATH}")
            print("   Using fallback (simple features)")
            self.yamnet = None
        else:
            self.yamnet = YAMNetTFLite(YAMNET_TFLITE_PATH)
        
        # Load audio classifier (your lung_audio.tflite)
        if not AUDIO_MODEL_PATH.exists():
            raise FileNotFoundError(f"Audio model not found: {AUDIO_MODEL_PATH}")
        
        self.classifier = tflite.Interpreter(str(AUDIO_MODEL_PATH))
        self.classifier.allocate_tensors()
        self.classifier_input = self.classifier.get_input_details()[0]
        self.classifier_output = self.classifier.get_output_details()[0]
        
        print("✅ Audio models loaded (TFLite only)")
    
    def _simple_audio_features(self, audio):
        """Fallback: simple MFCC-like features when YAMNet not available"""
        # Simple spectral features
        fft = np.abs(np.fft.rfft(audio, n=2048))[:1024]
        fft = fft / (np.max(fft) + 1e-6)
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / len(audio)
        
        # Energy
        energy = np.sum(audio ** 2) / len(audio)
        
        # Combine
        features = np.concatenate([fft[:100], [zcr, energy]])
        
        # Pad to 1024
        if len(features) < 1024:
            features = np.pad(features, (0, 1024 - len(features)))
        else:
            features = features[:1024]
        
        return features.astype(np.float32)
    
    def run(self):
        """Capture audio and run inference"""
        try:
            # Record audio
            recording = sd.rec(int(self.record_seconds * SAMPLE_RATE),
                               samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio = recording.flatten()
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Extract features
            if self.yamnet:
                # Use YAMNet to get 1024-dim embeddings
                embedding = self.yamnet.extract_embeddings(audio)
                input_data = embedding.reshape(1, -1).astype(np.float32)
            else:
                # Fallback to simple features
                input_data = self._simple_audio_features(audio).reshape(1, -1)
            
            # Run classifier
            self.classifier.set_tensor(self.classifier_input['index'], input_data)
            self.classifier.invoke()
            probs = self.classifier.get_tensor(self.classifier_output['index'])[0]
            
            self.result_ready.emit(probs.astype(np.float32))
            
            # Waveform for display
            if len(audio) > 800:
                down = audio[::len(audio)//800][:800]
                self.waveform_ready.emit(down)
                
        except Exception as e:
            print(f"Audio error: {e}")
            # Emit zeros on error
            self.result_ready.emit(np.zeros(5, dtype=np.float32))
        finally:
            self.finished.emit()

# =============================================================================
# RF Switch Controller
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
        print("✅ GPIO initialized for RF switches")
    
    def set_path(self, path_num):
        if path_num not in PATHS:
            raise ValueError(f"Invalid path: {path_num}")
        states = PATHS[path_num]
        for pin, state in states.items():
            GPIO.output(pin, state)
        time.sleep(0.05)  # Settle time
    
    def cleanup(self):
        GPIO.cleanup()
        print("GPIO cleaned up")

# =============================================================================
# VNA Network Client
# =============================================================================
class VNAClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connect()
    
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(10)
            print(f"✅ Connected to VNA server at {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"❌ VNA server connection failed: {e}")
            return False
    
    def capture_path(self):
        """Request VNA scan for current path"""
        if not self.socket:
            if not self.connect():
                return None
        
        try:
            self.socket.send(b'SCAN')
            response = self.socket.recv(65536).decode()
            data = json.loads(response)
            
            if data['status'] == 'success':
                return data['data']
            else:
                print(f"VNA error: {data.get('message', 'Unknown')}")
                return None
                
        except Exception as e:
            print(f"VNA communication error: {e}")
            self.socket = None
            return None
    
    def close(self):
        if self.socket:
            try:
                self.socket.send(b'QUIT')
                self.socket.close()
            except:
                pass

# =============================================================================
# Microwave Scanner
# =============================================================================
class MicrowaveScanner:
    def __init__(self, vna_client):
        self.vna = vna_client
        self.switch = RFSwitchController()
        self._baseline_data = None
        self._last_data = None
        self.frequencies = np.linspace(2e9, 3e9, 201)
    
    def scan_all_paths(self, progress_callback=None):
        """Scan all 4 antenna paths"""
        data = {}
        for path in [1, 2, 3, 4]:
            if progress_callback:
                progress_callback(f"Path {path}", path/4)
            
            # Set RF switch
            self.switch.set_path(path)
            
            # Capture via network VNA
            result = self.vna.capture_path()
            if result is None:
                raise RuntimeError(f"Failed to capture path {path}")
            
            # Extract magnitude dB array
            s21_db = np.array([point['mag_db'] for point in result])
            data[path] = s21_db
        
        self._last_data = data
        return data
    
    def extract_features(self, s21_data):
        """Convert S21 dict to 840-dim feature vector"""
        raw = np.array([s21_data[p] for p in [1,2,3,4]]).reshape(1, -1)
        return add_time_domain_features(raw)[0]
    
    def set_baseline(self, baseline_data):
        self._baseline_data = baseline_data
    
    def subtract_baseline(self, s21_data):
        if self._baseline_data is None:
            raise ValueError("No baseline stored")
        corrected = {}
        for p in [1,2,3,4]:
            air_linear = 10 ** (self._baseline_data[p] / 10)
            phantom_linear = 10 ** (s21_data[p] / 10)
            corrected_linear = phantom_linear - air_linear
            corrected[p] = 10 * np.log10(np.maximum(corrected_linear, 1e-12))
        return corrected
    
    def cleanup(self):
        self.switch.cleanup()

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
        
        print("✅ Fusion model loaded")
    
    def predict(self, mw_features, audio_probs):
        """Combine 840-dim microwave features + 5-dim audio probs"""
        fusion_vec = np.concatenate([mw_features, audio_probs]).reshape(1, -1)
        scaled = self.scaler.transform(fusion_vec)
        pred = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        return pred, np.max(proba)

# =============================================================================
# Main GUI Application
# =============================================================================
class ThoracisAIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("THORACIS‑AI: Lung Screening System")
        
        # Initialize components
        self.vna_client = VNAClient(VNA_SERVER_IP, VNA_SERVER_PORT)
        self.scanner = MicrowaveScanner(self.vna_client)
        
        try:
            self.fusion = FusionClassifier()
        except Exception as e:
            print(f"Fusion not loaded: {e}")
            self.fusion = None
        
        self.current_mw_features = None
        self.current_audio_probs = None
        self.current_s21_data = None
        self.baseline_data = None
        
        self._setup_ui()
        
        # IMPORTANT: Call showFullScreen AFTER all widgets are set up
        self.showFullScreen()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)   # Remove margins for full-screen
        layout.setSpacing(15)
        
        # Title
        title = QLabel("🫁 THORACIS‑AI")
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
        self.status_bar = QLabel(f"🌐 VNA Server: {VNA_SERVER_IP}:{VNA_SERVER_PORT}")
        self.status_bar.setStyleSheet("font-size: 12px; color: gray; padding: 5px;")
        layout.addWidget(self.status_bar)
        
        # IP input for reconnection
        ip_layout = QHBoxLayout()
        ip_label = QLabel("Server IP:")
        self.ip_input = QLineEdit(VNA_SERVER_IP)
        self.ip_input.setMaximumWidth(150)
        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.clicked.connect(self._reconnect_vna)
        ip_layout.addWidget(ip_label)
        ip_layout.addWidget(self.ip_input)
        ip_layout.addWidget(self.reconnect_btn)
        ip_layout.addStretch()
        layout.addLayout(ip_layout)
        
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
        
        # Add tabs
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
        layout.setContentsMargins(0, 0, 0, 0)   # Remove margins
        layout.setSpacing(10)
        
        self.mw_status = QLabel("✅ Ready")
        self.mw_status.setStyleSheet("font-size: 14px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.mw_status)
        
        self.baseline_btn = QPushButton("1️⃣ RECORD BASELINE (AIR)")
        self.baseline_btn.setMinimumHeight(60)
        self.baseline_btn.setStyleSheet(self._button_style("#81d4fa"))
        self.baseline_btn.clicked.connect(self._record_baseline)
        layout.addWidget(self.baseline_btn)
        
        self.scan_btn = QPushButton("2️⃣ SCAN PHANTOM / PATIENT")
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
        
        self.reconstruct_btn = QPushButton("📊 RECONSTRUCT IMAGE")
        self.reconstruct_btn.setEnabled(False)
        self.reconstruct_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.reconstruct_btn.clicked.connect(self._show_reconstruction)
        layout.addWidget(self.reconstruct_btn)
        
        self.tabs.addTab(tab, "📡 Microwave Scan")
    
    def _add_audio_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)   # Remove margins
        layout.setSpacing(10)
        
        self.audio_status = QLabel("✅ Ready")
        self.audio_status.setStyleSheet("font-size: 14px; padding: 8px; background: #e8f5e9; border-radius: 8px;")
        layout.addWidget(self.audio_status)
        
        self.waveform_label = QLabel()
        self.waveform_label.setMinimumHeight(150)
        self.waveform_label.setStyleSheet("background-color: black; border-radius: 8px;")
        layout.addWidget(self.waveform_label)
        
        self.audio_btn = QPushButton("🎤 ANALYZE LUNG SOUNDS")
        self.audio_btn.setMinimumHeight(80)
        self.audio_btn.setStyleSheet(self._button_style("#66bb6a"))
        self.audio_btn.clicked.connect(self._run_acoustic_analysis)
        layout.addWidget(self.audio_btn)
        
        self.audio_result = QTextEdit()
        self.audio_result.setReadOnly(True)
        self.audio_result.setMinimumHeight(150)
        layout.addWidget(self.audio_result)
        
        self.tabs.addTab(tab, "🎧 Acoustic Analysis")
    
    def _add_fusion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)   # Remove margins
        layout.setSpacing(10)
        
        self.fusion_status = QLabel("Perform both scans for combined diagnosis")
        self.fusion_status.setStyleSheet("font-size: 14px; padding: 8px; background: #fff3e0; border-radius: 8px;")
        layout.addWidget(self.fusion_status)
        
        self.fusion_mw_btn = QPushButton("1️⃣ SCAN MICROWAVE")
        self.fusion_mw_btn.setMinimumHeight(60)
        self.fusion_mw_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_mw_btn.clicked.connect(self._fusion_microwave)
        layout.addWidget(self.fusion_mw_btn)
        
        self.fusion_audio_btn = QPushButton("2️⃣ ANALYZE ACOUSTIC")
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_audio_btn.setMinimumHeight(60)
        self.fusion_audio_btn.setStyleSheet(self._button_style("#ffb74d"))
        self.fusion_audio_btn.clicked.connect(self._fusion_acoustic)
        layout.addWidget(self.fusion_audio_btn)
        
        self.fusion_combine_btn = QPushButton("3️⃣ RUN FUSION DIAGNOSIS")
        self.fusion_combine_btn.setEnabled(False)
        self.fusion_combine_btn.setMinimumHeight(80)
        self.fusion_combine_btn.setStyleSheet(self._button_style("#4fc3f7"))
        self.fusion_combine_btn.clicked.connect(self._run_fusion)
        layout.addWidget(self.fusion_combine_btn)
        
        self.fusion_result = QTextEdit()
        self.fusion_result.setReadOnly(True)
        self.fusion_result.setMinimumHeight(150)
        layout.addWidget(self.fusion_result)
        
        self.tabs.addTab(tab, "🤝 Full Fusion")
    
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
    
    def _reconnect_vna(self):
        new_ip = self.ip_input.text()
        self.vna_client.close()
        self.vna_client = VNAClient(new_ip, VNA_SERVER_PORT)
        self.status_bar.setText(f"🌐 VNA Server: {new_ip}:{VNA_SERVER_PORT}")
    
    def _record_baseline(self):
        self.baseline_btn.setEnabled(False)
        self.mw_status.setText("📡 Recording baseline (air)...")
        self.mw_progress.setVisible(True)
        
        def worker():
            try:
                data = self.scanner.scan_all_paths(
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.scanner.set_baseline(data)
                self.baseline_data = data
                self.mw_result.setText("✅ Baseline recorded successfully!\n\nNow place phantom/patient and click SCAN.")
            except Exception as e:
                self.mw_result.setText(f"❌ Error: {e}")
            finally:
                self.baseline_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _run_microwave_scan(self):
        if self.scanner._baseline_data is None:
            QMessageBox.warning(self, "Missing Baseline", "Please record baseline (air) first!")
            return
        
        self.scan_btn.setEnabled(False)
        self.mw_progress.setVisible(True)
        self.mw_result.setText("📡 Scanning...")
        
        def worker():
            try:
                data = self.scanner.scan_all_paths(
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.current_s21_data = data
                corrected = self.scanner.subtract_baseline(data)
                features = self.scanner.extract_features(corrected)
                self.current_mw_features = features
                
                # Calculate average S21 for quick feedback
                avg_s21 = np.mean([np.mean(data[p]) for p in [1,2,3,4]])
                self.mw_result.setText(f"✅ Scan complete!\n\nAverage S21: {avg_s21:.1f} dB\nBackground subtracted.\nReady for fusion.")
                self.reconstruct_btn.setEnabled(True)
            except Exception as e:
                self.mw_result.setText(f"❌ Error: {e}")
            finally:
                self.scan_btn.setEnabled(True)
                self.mw_progress.setVisible(False)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _update_mw_progress(self, msg, frac):
        self.mw_status.setText(f"📡 {msg}")
        self.mw_progress.setValue(int(frac * 100))
        QApplication.processEvents()
    
    def _show_reconstruction(self):
        if self.current_s21_data is None:
            QMessageBox.information(self, "No Data", "Perform a scan first.")
            return
        
        # Simple reconstruction display
        s21_values = [np.mean(self.current_s21_data[p]) for p in [1,2,3,4]]
        msg = f"📊 Path-wise S21 averages:\n\n"
        msg += f"Path 1 (1→3): {s21_values[0]:.1f} dB\n"
        msg += f"Path 2 (1→4): {s21_values[1]:.1f} dB\n"
        msg += f"Path 3 (2→3): {s21_values[2]:.1f} dB\n"
        msg += f"Path 4 (2→4): {s21_values[3]:.1f} dB\n\n"
        msg += f"Spatial asymmetry: {abs(s21_values[0] - s21_values[3]):.1f} dB\n"
        msg += f"Tumor indicator: {'Positive' if max(s21_values) - min(s21_values) > 5 else 'Negative'}"
        
        QMessageBox.information(self, "Microwave Analysis", msg)
    
    def _run_acoustic_analysis(self):
        self.audio_btn.setEnabled(False)
        self.audio_status.setText("🎤 Recording (3 seconds)...")
        self.audio_result.setText("")
        
        self.audio_thread = AcousticProcessor(RECORD_SECONDS)
        self.audio_thread.result_ready.connect(self._on_audio_result)
        self.audio_thread.waveform_ready.connect(self._draw_waveform)
        self.audio_thread.finished.connect(self._on_audio_finished)
        self.audio_thread.start()
    
    def _on_audio_result(self, probs):
        self.current_audio_probs = probs
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        class_name = AUDIO_CLASSES[class_idx]
        
        self.audio_result.setText(
            f"🎯 PREDICTION: {class_name.upper()}\n"
            f"📊 Confidence: {confidence:.1%}\n\n"
            f"📈 Detailed Probabilities:\n" +
            "\n".join(f"   {c}: {p:.1%}" for c, p in zip(AUDIO_CLASSES, probs))
        )
        self.audio_status.setText(f"✅ Result: {class_name}")
    
    def _on_audio_finished(self):
        self.audio_btn.setEnabled(True)
    
    def _draw_waveform(self, data):
        if data is None or len(data) == 0:
            return
        w = self.waveform_label.width()
        h = self.waveform_label.height()
        if w < 10:
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
        self.fusion_mw_btn.setEnabled(False)
        self.fusion_status.setText("📡 Scanning microwave...")
        
        def worker():
            try:
                if self.scanner._baseline_data is None:
                    self.fusion_status.setText("Need baseline first!")
                    self.fusion_mw_btn.setEnabled(True)
                    return
                
                data = self.scanner.scan_all_paths()
                corrected = self.scanner.subtract_baseline(data)
                features = self.scanner.extract_features(corrected)
                self.current_mw_features = features
                self.fusion_status.setText("✅ Microwave complete. Now perform acoustic.")
                self.fusion_audio_btn.setEnabled(True)
            except Exception as e:
                self.fusion_status.setText(f"❌ Error: {e}")
            finally:
                self.fusion_mw_btn.setEnabled(True)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _fusion_acoustic(self):
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_status.setText("🎤 Recording acoustic...")
        self.audio_thread = AcousticProcessor(RECORD_SECONDS)
        self.audio_thread.result_ready.connect(self._on_fusion_audio_result)
        self.audio_thread.finished.connect(self._on_fusion_audio_finished)
        self.audio_thread.start()
    
    def _on_fusion_audio_result(self, probs):
        self.current_audio_probs = probs
        self.fusion_status.setText("✅ Acoustic complete. Ready for fusion.")
        self.fusion_combine_btn.setEnabled(True)
    
    def _on_fusion_audio_finished(self):
        self.fusion_audio_btn.setEnabled(True)
    
    def _run_fusion(self):
        if self.current_mw_features is None or self.current_audio_probs is None:
            QMessageBox.warning(self, "Missing Data", "Perform both scans first!")
            return
        
        if self.fusion is None:
            self.fusion_result.setText("❌ Fusion model not loaded!")
            return
        
        try:
            pred, conf = self.fusion.predict(self.current_mw_features, self.current_audio_probs)
            class_names = ['baseline', 'healthy', 'tumor']
            result = class_names[pred] if pred < 3 else "unknown"
            audio_class = AUDIO_CLASSES[np.argmax(self.current_audio_probs)]
            
            self.fusion_result.setText(
                f"🫁 FUSION DIAGNOSIS: {result.upper()}\n\n"
                f"📊 Confidence: {conf:.1%}\n\n"
                f"🔬 Interpretation:\n"
                f"   • Structural anomaly: {['None', 'Possible', 'Definite'][pred]}\n"
                f"   • Functional pattern: {audio_class}\n\n"
                f"💡 Clinical Note:\n"
                f"   {'Normal findings' if result == 'baseline' else 'Further evaluation recommended'}"
            )
            self.fusion_status.setText(f"✅ Diagnosis: {result}")
        except Exception as e:
            self.fusion_result.setText(f"❌ Fusion error: {e}")
    
    def closeEvent(self, event):
        print("\n🛑 Shutting down THORACIS-AI...")
        self.scanner.cleanup()
        self.vna_client.close()
        event.accept()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import os

    # Only set OpenGL ES attribute if we are actually using eglfs
    platform = os.environ.get('QT_QPA_PLATFORM', '')
    if platform == 'eglfs':
        QApplication.setAttribute(Qt.AA_UseOpenGLES, True)

    app = QApplication(sys.argv)
    # HighDPI scaling is not needed for linuxfb, and may cause issues; removed.

    window = ThoracisoAIMainWindow()
    # Optionally force the window to fill the screen (though showFullScreen should handle it)
    # screen = QApplication.primaryScreen()
    # window.setGeometry(0, 0, screen.size().width(), screen.size().height())
    window.showFullScreen()   # already called in constructor, but safe to call again

    sys.exit(app.exec())
