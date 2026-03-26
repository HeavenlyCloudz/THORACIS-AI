#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULMO‑AI Unified Application – Final Version
- Microwave scanning (4 paths)
- Acoustic analysis using YAMNet + your audio model (5 classes)
- Fusion classification using XGBoost
- Image reconstruction with background subtraction
- Touch‑friendly GUI (blue theme)
"""

import sys
import os
import time
import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle

# Qt
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QProgressBar, QMessageBox, QTabWidget, QTextEdit
)

# Hardware / communication
import serial
import RPi.GPIO as GPIO
import sounddevice as sd

# ML
import tflite_runtime.interpreter as tflite
import tensorflow as tf  # needed for YAMNet (can be large, but we'll load only once)
import tensorflow_hub as hub

# Local modules (place these files next to this script)
from image_reconstructor import MicrowaveImageReconstructor
from feature_extractor import FeatureExtractor   # optional, for extra features

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path.home() / "pulmo_ai_app" / "models"
AUDIO_MODEL_PATH = MODEL_DIR / "lung_audio.tflite"
MICROWAVE_MODEL_PATH = MODEL_DIR / "xgboost_final_model.pkl"
MICROWAVE_SCALER_PATH = MODEL_DIR / "xgboost_final_scaler.pkl"
FUSION_MODEL_PATH = MODEL_DIR / "fusion_xgboost_model.pkl"
FUSION_SCALER_PATH = MODEL_DIR / "fusion_scaler.pkl"

# VNA settings
VNA_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
START_FREQ = 2e9
STOP_FREQ = 3e9
FREQ_POINTS = 201

# GPIO pins (RF switches)
SWITCH1_A = 17   # TX: antenna 1
SWITCH1_B = 27   # TX: antenna 2
SWITCH2_A = 18   # RX: antenna 3
SWITCH2_B = 22   # RX: antenna 4

PATHS = {
    1: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 1, SWITCH2_B: 0},  # 1→3
    2: {SWITCH1_A: 1, SWITCH1_B: 0, SWITCH2_A: 0, SWITCH2_B: 1},  # 1→4
    3: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 1, SWITCH2_B: 0},  # 2→3
    4: {SWITCH1_A: 0, SWITCH1_B: 1, SWITCH2_A: 0, SWITCH2_B: 1},  # 2→4
}

# Audio settings
SAMPLE_RATE = 16000          # YAMNet expects 16kHz
RECORD_SECONDS = 3
YAMNET_PATH = "https://tfhub.dev/google/yamnet/1"  # or local tflite

# Fusion class mapping (microwave -> audio class)
AUDIO_CLASSES = ['asthma', 'copd', 'pneumonia', 'healthy', 'Bronchial']

# =============================================================================
# TIME‑DOMAIN FEATURES (matches training)
# =============================================================================
def add_time_domain_features(X):
    """Add IFFT‑based time‑domain features (840 total)."""
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
                np.max(time_resp),
                np.argmax(time_resp),
                np.mean(time_resp),
                np.std(time_resp),
                np.percentile(time_resp, 90),
                np.percentile(time_resp, 10),
                np.sum(time_resp),
                np.max(time_resp) - np.min(time_resp),
                np.sum(time_resp ** 2)
            ])
        time_features.append(sample_time)
    time_features = np.array(time_features)
    return np.concatenate([X, time_features], axis=1)

# =============================================================================
# MICROWAVE SCANNER
# =============================================================================
class MicrowaveScanner:
    def __init__(self):
        self.frequencies = np.linspace(START_FREQ, STOP_FREQ, FREQ_POINTS)
        self._setup_gpio()
        self._last_data = None
        self._baseline_data = None   # for background subtraction

    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in [SWITCH1_A, SWITCH1_B, SWITCH2_A, SWITCH2_B]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)

    def _set_path(self, path_num):
        states = PATHS.get(path_num)
        if not states:
            raise ValueError(f"Invalid path: {path_num}")
        for pin, state in states.items():
            GPIO.output(pin, state)
        time.sleep(0.05)

    def _capture_single_path(self, path_num):
        self._set_path(path_num)
        with serial.Serial(VNA_PORT, BAUDRATE, timeout=2) as ser:
            time.sleep(0.5)
            ser.reset_input_buffer()
            cmd = f"scan {START_FREQ} {STOP_FREQ} {FREQ_POINTS} 5"
            ser.write((cmd + '\n').encode())
            time.sleep(1.5)
            s21_db = []
            timeout = time.time() + 15
            while len(s21_db) < FREQ_POINTS and time.time() < timeout:
                line = ser.readline().decode('ascii', errors='ignore').strip()
                if line and not line.startswith(('ch>', 'scan')):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            real = float(parts[1]); imag = float(parts[2])
                            mag = np.sqrt(real*real + imag*imag)
                            mag_db = 20 * np.log10(mag) if mag > 0 else -120
                            s21_db.append(mag_db)
                        except:
                            continue
            if len(s21_db) != FREQ_POINTS:
                raise RuntimeError(f"Only {len(s21_db)}/{FREQ_POINTS} points")
            return np.array(s21_db)

    def scan_all_paths(self, progress_callback=None):
        data = {}
        for path in [1,2,3,4]:
            if progress_callback:
                progress_callback(f"Path {path}", path/4)
            data[path] = self._capture_single_path(path)
        self._last_data = data
        return data

    def extract_features(self, s21_data):
        """Return 840‑dim feature vector (804 freq + 36 time)."""
        raw = np.array([s21_data[p] for p in [1,2,3,4]]).reshape(1, -1)
        return add_time_domain_features(raw)[0]

    def set_baseline(self, baseline_data):
        """Store baseline (air) for background subtraction."""
        self._baseline_data = baseline_data

    def subtract_baseline(self, s21_data):
        """Return background‑subtracted S21 dict."""
        if self._baseline_data is None:
            raise ValueError("No baseline stored. Perform baseline scan first.")
        corrected = {}
        for p in [1,2,3,4]:
            air_linear = 10**(self._baseline_data[p] / 10)
            phantom_linear = 10**(s21_data[p] / 10)
            corrected_linear = phantom_linear - air_linear
            corrected[p] = 10 * np.log10(np.maximum(corrected_linear, 1e-12))
        return corrected

    def cleanup(self):
        GPIO.cleanup()

# =============================================================================
# ACOUSTIC PROCESSOR (using YAMNet + your audio model)
# =============================================================================
class AcousticProcessor(QThread):
    result_ready = Signal(np.ndarray)   # sends 5‑class probabilities
    waveform_ready = Signal(np.ndarray)
    finished = Signal()

    def __init__(self, record_seconds=3):
        super().__init__()
        self.record_seconds = record_seconds
        self.sample_rate = SAMPLE_RATE
        self._load_models()
        self._stop_flag = False

    def _load_models(self):
        # 1. Load YAMNet (from TF Hub or local)
        try:
            self.yamnet = hub.load(YAMNET_PATH)
            print("YAMNet loaded from TF Hub")
        except:
            # Fallback: use local tflite if available
            yamnet_tflite = MODEL_DIR / "yamnet.tflite"
            if yamnet_tflite.exists():
                self.yamnet_interpreter = tflite.Interpreter(str(yamnet_tflite))
                self.yamnet_interpreter.allocate_tensors()
                self.yamnet_input = self.yamnet_interpreter.get_input_details()[0]
                self.yamnet_output = self.yamnet_interpreter.get_output_details()[0]
                self.use_tflite_yamnet = True
                print("YAMNet loaded from local TFLite")
            else:
                raise RuntimeError("No YAMNet model found")

        # 2. Load your audio classifier (expects 1024‑dim embedding)
        self.audio_classifier = tflite.Interpreter(str(AUDIO_MODEL_PATH))
        self.audio_classifier.allocate_tensors()
        self.classifier_input = self.audio_classifier.get_input_details()[0]
        self.classifier_output = self.audio_classifier.get_output_details()[0]
        print("Audio classifier loaded")

    def _extract_yamnet_embedding(self, audio):
        """Convert raw audio (16kHz) to 1024‑dim embedding (mean over time)."""
        if hasattr(self, 'use_tflite_yamnet'):
            # TFLite version
            self.yamnet_interpreter.set_tensor(self.yamnet_input['index'], audio.reshape(1, -1))
            self.yamnet_interpreter.invoke()
            embeddings = self.yamnet_interpreter.get_tensor(self.yamnet_output['index'])
            # embeddings shape: (num_frames, 1024)
            return np.mean(embeddings, axis=0)
        else:
            # TF Hub version
            scores, embeddings, _ = self.yamnet(audio)
            return np.mean(embeddings.numpy(), axis=0)

    def run(self):
        try:
            # Record
            recording = sd.rec(int(self.record_seconds * self.sample_rate),
                               samplerate=self.sample_rate, channels=1, dtype='float32')
            sd.wait()
            audio = recording.flatten()

            # Resample if needed (already 16kHz)
            if self.sample_rate != 16000:
                # simple resampling placeholder
                pass

            # Ensure minimum length (YAMNet needs 0.96s)
            min_len = int(0.96 * 16000)
            if len(audio) < min_len:
                audio = np.pad(audio, (0, min_len - len(audio)))
            elif len(audio) > 3 * 16000:
                audio = audio[:3 * 16000]

            # Extract YAMNet embedding
            embedding = self._extract_yamnet_embedding(audio.astype(np.float32))
            # embedding shape: (1024,)
            input_emb = embedding.reshape(1, -1).astype(np.float32)

            # Run classifier
            self.audio_classifier.set_tensor(self.classifier_input['index'], input_emb)
            self.audio_classifier.invoke()
            probs = self.audio_classifier.get_tensor(self.classifier_output['index'])[0]  # (5,)

            # Emit results
            self.result_ready.emit(probs)

            # Downsample waveform for display
            if len(audio) > 800:
                down = audio[::len(audio)//800][:800]
                self.waveform_ready.emit(down)

        except Exception as e:
            print(f"Acoustic error: {e}")
            # Emit zeros on error
            self.result_ready.emit(np.zeros(5))
        finally:
            self.finished.emit()

# =============================================================================
# FUSION CLASSIFIER
# =============================================================================
class FusionClassifier:
    def __init__(self):
        with open(FUSION_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(FUSION_SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict(self, mw_features, audio_probs):
        """mw_features: 840‑dim; audio_probs: 5‑dim; returns (pred_class, confidence)."""
        fusion_vec = np.concatenate([mw_features, audio_probs]).reshape(1, -1)
        scaled = self.scaler.transform(fusion_vec)
        pred = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        return pred, np.max(proba)

# =============================================================================
# MAIN GUI APPLICATION
# =============================================================================
class PulmoAIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PULMO‑AI: Lung Screening System")
        self.showFullScreen()

        self.scanner = MicrowaveScanner()
        self.fusion = None
        try:
            self.fusion = FusionClassifier()
        except Exception as e:
            print(f"Fusion not loaded: {e}")

        self.reconstructor = MicrowaveImageReconstructor()
        self.current_mw_features = None
        self.current_audio_probs = None
        self.current_s21_data = None
        self.baseline_data = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title (blue theme)
        title = QLabel("PULMO‑AI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #0277bd;
            background-color: #e1f5fe;
            padding: 20px;
            border-radius: 20px;
        """)
        main_layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(self._tab_style())
        main_layout.addWidget(self.tabs)

        # Tab 1: Microwave Scan
        self.mw_tab = QWidget()
        mw_layout = QVBoxLayout(self.mw_tab)
        self.mw_status = QLabel("Ready")
        self.mw_status.setStyleSheet("font-size: 16px; padding: 10px;")
        mw_layout.addWidget(self.mw_status)

        self.baseline_btn = QPushButton("1. RECORD BASELINE (AIR)")
        self.baseline_btn.setMinimumHeight(60)
        self.baseline_btn.setStyleSheet(self._button_style("#81d4fa"))
        mw_layout.addWidget(self.baseline_btn)

        self.mw_scan_btn = QPushButton("2. SCAN PHANTOM / PATIENT")
        self.mw_scan_btn.setMinimumHeight(80)
        self.mw_scan_btn.setStyleSheet(self._button_style("#4fc3f7"))
        mw_layout.addWidget(self.mw_scan_btn)

        self.mw_progress = QProgressBar()
        self.mw_progress.setVisible(False)
        mw_layout.addWidget(self.mw_progress)

        self.mw_result = QTextEdit()
        self.mw_result.setReadOnly(True)
        self.mw_result.setMinimumHeight(200)
        mw_layout.addWidget(self.mw_result)

        self.image_btn = QPushButton("RECONSTRUCT IMAGE")
        self.image_btn.setEnabled(False)
        self.image_btn.setStyleSheet(self._button_style("#ffb74d"))
        mw_layout.addWidget(self.image_btn)

        mw_layout.addStretch()
        self.tabs.addTab(self.mw_tab, "Microwave Scan")

        # Tab 2: Acoustic Analysis
        self.audio_tab = QWidget()
        audio_layout = QVBoxLayout(self.audio_tab)
        self.audio_status = QLabel("Ready")
        self.audio_status.setStyleSheet("font-size: 16px; padding: 10px;")
        audio_layout.addWidget(self.audio_status)

        self.waveform_label = QLabel()
        self.waveform_label.setMinimumHeight(150)
        self.waveform_label.setStyleSheet("background-color: black; border-radius: 8px;")
        audio_layout.addWidget(self.waveform_label)

        self.audio_btn = QPushButton("ANALYZE LUNG SOUNDS")
        self.audio_btn.setMinimumHeight(80)
        self.audio_btn.setStyleSheet(self._button_style("#66bb6a"))
        audio_layout.addWidget(self.audio_btn)

        self.audio_result = QTextEdit()
        self.audio_result.setReadOnly(True)
        self.audio_result.setMinimumHeight(150)
        audio_layout.addWidget(self.audio_result)

        audio_layout.addStretch()
        self.tabs.addTab(self.audio_tab, "Acoustic Analysis")

        # Tab 3: Full Fusion
        self.fusion_tab = QWidget()
        fusion_layout = QVBoxLayout(self.fusion_tab)
        self.fusion_status = QLabel("Perform both scans for combined diagnosis")
        self.fusion_status.setStyleSheet("font-size: 16px; padding: 10px;")
        fusion_layout.addWidget(self.fusion_status)

        self.fusion_mw_btn = QPushButton("1. SCAN MICROWAVE")
        self.fusion_mw_btn.setMinimumHeight(60)
        self.fusion_mw_btn.setStyleSheet(self._button_style("#ffb74d"))
        fusion_layout.addWidget(self.fusion_mw_btn)

        self.fusion_audio_btn = QPushButton("2. ANALYZE ACOUSTIC")
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_audio_btn.setStyleSheet(self._button_style("#ffb74d"))
        fusion_layout.addWidget(self.fusion_audio_btn)

        self.fusion_combine_btn = QPushButton("3. RUN FUSION DIAGNOSIS")
        self.fusion_combine_btn.setEnabled(False)
        self.fusion_combine_btn.setMinimumHeight(80)
        self.fusion_combine_btn.setStyleSheet(self._button_style("#4fc3f7"))
        fusion_layout.addWidget(self.fusion_combine_btn)

        self.fusion_result = QTextEdit()
        self.fusion_result.setReadOnly(True)
        self.fusion_result.setMinimumHeight(200)
        fusion_layout.addWidget(self.fusion_result)

        fusion_layout.addStretch()
        self.tabs.addTab(self.fusion_tab, "Full Fusion")

        # Exit button
        exit_btn = QPushButton("EXIT")
        exit_btn.setMinimumHeight(60)
        exit_btn.setStyleSheet(self._button_style("#ef5350"))
        exit_btn.clicked.connect(self.close)
        main_layout.addWidget(exit_btn)

    def _tab_style(self):
        return """
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
        """

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

    def _connect_signals(self):
        self.baseline_btn.clicked.connect(self._record_baseline)
        self.mw_scan_btn.clicked.connect(self._run_microwave_scan)
        self.image_btn.clicked.connect(self._show_reconstructed_image)
        self.audio_btn.clicked.connect(self._run_acoustic_analysis)
        self.fusion_mw_btn.clicked.connect(self._fusion_microwave_step)
        self.fusion_audio_btn.clicked.connect(self._fusion_acoustic_step)
        self.fusion_combine_btn.clicked.connect(self._run_fusion)

    # --- Microwave Tab ---
    def _record_baseline(self):
        """Record baseline (air) for later subtraction."""
        self.baseline_btn.setEnabled(False)
        self.mw_status.setText("Recording baseline (air)...")
        self.mw_progress.setVisible(True)
        try:
            data = self.scanner.scan_all_paths(
                progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
            )
            self.scanner.set_baseline(data)
            self.baseline_data = data
            self.mw_result.setText("Baseline recorded successfully.\nNow place phantom and scan.")
            self.mw_status.setText("Baseline ready")
        except Exception as e:
            self.mw_result.setText(f"Error: {e}")
        finally:
            self.baseline_btn.setEnabled(True)
            self.mw_progress.setVisible(False)

    def _run_microwave_scan(self):
        if self.scanner._baseline_data is None:
            QMessageBox.warning(self, "Missing Baseline",
                                "Please record baseline (air) first.")
            return
        self.mw_scan_btn.setEnabled(False)
        self.mw_progress.setVisible(True)
        self.mw_result.setText("Scanning...")

        def worker():
            try:
                data = self.scanner.scan_all_paths(
                    progress_callback=lambda msg, f: self._update_mw_progress(msg, f)
                )
                self.current_s21_data = data
                # Background subtraction
                corrected = self.scanner.subtract_baseline(data)
                features = self.scanner.extract_features(corrected)
                self.current_mw_features = features
                self.mw_result.setText("Scan complete.\nBackground subtracted.\nReady for fusion.")
                self.image_btn.setEnabled(True)
            except Exception as e:
                self.mw_result.setText(f"Error: {e}")
            finally:
                self.mw_progress.setVisible(False)
                self.mw_scan_btn.setEnabled(True)

        threading.Thread(target=worker, daemon=True).start()

    def _update_mw_progress(self, msg, frac):
        self.mw_status.setText(msg)
        self.mw_progress.setValue(int(frac * 100))
        QApplication.processEvents()

    def _show_reconstructed_image(self):
        if self.current_s21_data is None:
            QMessageBox.information(self, "No Data", "Perform a scan first.")
            return
        try:
            # Prepare measurements for reconstructor (needs list of dicts)
            measurements = []
            for path in [1,2,3,4]:
                meas = {
                    'tx_antenna': 1 if path in (1,2) else 2,
                    'rx_antenna': 3 if path in (1,3) else 4,
                    'mean_magnitude_db': np.mean(self.current_s21_data[path]),
                    's21_magnitudes_db': self.current_s21_data[path],
                    's21_phases_deg': [0]*FREQ_POINTS,  # phase not used here
                    'timestamp': time.time(),
                    'scan_label': 'phantom'
                }
                measurements.append(meas)

            # Use the reconstructor
            fig = self.reconstructor.plot_reconstruction(measurements, title="PULMO‑AI Reconstruction")
            # Show in a new window (Qt)
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            canvas = FigureCanvas(fig)
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Microwave Image Reconstruction")
            layout = QVBoxLayout(dialog)
            layout.addWidget(canvas)
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            dialog.exec_()
            plt.close(fig)
        except Exception as e:
            QMessageBox.warning(self, "Reconstruction Error", str(e))

    # --- Acoustic Tab ---
    def _run_acoustic_analysis(self):
        self.audio_btn.setEnabled(False)
        self.audio_status.setText("Recording...")
        self.audio_result.setText("")
        self.acoustic_thread = AcousticProcessor(RECORD_SECONDS)
        self.acoustic_thread.result_ready.connect(self._on_audio_result)
        self.acoustic_thread.waveform_ready.connect(self._draw_waveform)
        self.acoustic_thread.finished.connect(self._on_audio_finished)
        self.acoustic_thread.start()

    def _on_audio_result(self, probs):
        self.current_audio_probs = probs
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        class_name = AUDIO_CLASSES[class_idx]
        self.audio_result.setText(
            f"Acoustic analysis complete.\n"
            f"Prediction: {class_name.upper()}\n"
            f"Confidence: {confidence:.1%}\n\n"
            f"Probabilities:\n" +
            "\n".join(f"  {c}: {p:.1%}" for c,p in zip(AUDIO_CLASSES, probs))
        )

    def _on_audio_finished(self):
        self.audio_btn.setEnabled(True)
        self.audio_status.setText("Ready")

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

    # --- Fusion Tab ---
    def _fusion_microwave_step(self):
        self.fusion_mw_btn.setEnabled(False)
        self.fusion_status.setText("Scanning microwave...")
        self.fusion_result.setText("")
        if self.scanner._baseline_data is None:
            QMessageBox.warning(self, "Missing Baseline", "Please record baseline first.")
            self.fusion_mw_btn.setEnabled(True)
            return

        def worker():
            try:
                data = self.scanner.scan_all_paths()
                corrected = self.scanner.subtract_baseline(data)
                features = self.scanner.extract_features(corrected)
                self.current_mw_features = features
                self.fusion_status.setText("Microwave scan complete. Now perform acoustic.")
                self.fusion_audio_btn.setEnabled(True)
            except Exception as e:
                self.fusion_status.setText(f"Error: {str(e)}")
            finally:
                self.fusion_mw_btn.setEnabled(True)

        threading.Thread(target=worker, daemon=True).start()

    def _fusion_acoustic_step(self):
        self.fusion_audio_btn.setEnabled(False)
        self.fusion_status.setText("Recording acoustic...")
        self.acoustic_thread = AcousticProcessor(RECORD_SECONDS)
        self.acoustic_thread.result_ready.connect(self._on_fusion_audio_result)
        self.acoustic_thread.finished.connect(self._on_fusion_audio_finished)
        self.acoustic_thread.start()

    def _on_fusion_audio_result(self, probs):
        self.current_audio_probs = probs
        self.fusion_status.setText("Acoustic complete. Ready for fusion.")
        self.fusion_combine_btn.setEnabled(True)

    def _on_fusion_audio_finished(self):
        self.fusion_audio_btn.setEnabled(True)

    def _run_fusion(self):
        if self.current_mw_features is None or self.current_audio_probs is None:
            QMessageBox.warning(self, "Missing Data", "Perform both scans first.")
            return
        try:
            pred_class, confidence = self.fusion.predict(
                self.current_mw_features, self.current_audio_probs
            )
            class_names = ['baseline', 'healthy', 'tumor']
            result = class_names[pred_class] if pred_class < 3 else "unknown"
            audio_class = AUDIO_CLASSES[np.argmax(self.current_audio_probs)]
            self.fusion_result.setText(
                f"FUSION DIAGNOSIS:\n\n{result.upper()}\n\n"
                f"Confidence: {confidence:.1%}\n\n"
                f"Interpretation:\n"
                f"- Structural anomaly: {['No', 'Possible', 'Definite'][pred_class]}\n"
                f"- Functional pattern: {audio_class}"
            )
            self.fusion_status.setText("Diagnosis complete.")
        except Exception as e:
            self.fusion_result.setText(f"Fusion error: {str(e)}")

    def closeEvent(self, event):
        self.scanner.cleanup()
        event.accept()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    # Qt for eglfs (fullscreen on Pi)
    os.environ['QT_QPA_PLATFORM'] = 'eglfs'
    os.environ['QT_QPA_EGLFS_HIDECURSOR'] = '1'

    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    window = PulmoAIMainWindow()
    window.show()

    sys.exit(app.exec())
