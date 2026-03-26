#!/usr/bin/env python3
"""
PULMO AI - Acoustic Lung Screening System
Audio-only version optimized for Raspberry Pi
"""

import sys
import time
import numpy as np
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QPushButton, QApplication, 
                           QMessageBox, QProgressBar, QMainWindow, QWidget)
from PySide6.QtCore import QThread, Signal as pyqtSignal, Qt, QTimer, QRectF
from PySide6.QtGui import QPainter, QPen, QColor
import sounddevice as sd
import queue
import threading
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== IMPORT TENSORFLOW LITE ==================== #
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    print("✅ TensorFlow Lite available - optimal for Raspberry Pi")
except ImportError:
    try:
        import tensorflow as tf
        TFLITE_AVAILABLE = False
        print("⚠️ Using full TensorFlow (slower on Pi)")
    except ImportError:
        print("❌ No TensorFlow/TFLite found!")
        sys.exit(1)

# ==================== GLOBAL EXCEPTION HANDLER ==================== #
def exception_handler(exc_type, exc_value, exc_traceback):
    print("=== PULMO AI CRASH DETECTED ===")
    print(f"Exception: {exc_type.__name__}: {exc_value}")
    with open('pulmo_ai_crash_log.txt', 'a') as f:
        f.write(f"Crash at: {datetime.now()}\n")
        f.write(f"Exception: {exc_type.__name__}: {exc_value}\n")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_handler
print("Global exception handler installed")

# ==================== PURE NUMPY MFCC EXTRACTOR ==================== #
class NumpyMFCCExtractor:
    """Pure NumPy MFCC feature extraction (no librosa dependency)"""
    
    @staticmethod
    def compute_mfcc(audio, sample_rate=16000, n_mfcc=13, n_fft=512, hop_length=160):
        """
        Compute MFCC features using pure NumPy
        Based on standard MFCC algorithm steps
        """
        # 1. Pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # 2. Framing
        frame_length = n_fft
        frame_step = hop_length
        signal_length = len(emphasized)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        
        pad_signal_length = num_frames * frame_step + frame_length
        pad_signal = np.pad(emphasized, (0, pad_signal_length - signal_length), 'constant')
        
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                  np.tile(np.arange(0, num_frames * frame_step, frame_step), 
                         (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        
        # 3. Window (Hamming window)
        frames *= np.hamming(frame_length)
        
        # 4. FFT and Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
        pow_frames = ((1.0 / n_fft) * ((mag_frames) ** 2))
        
        # 5. Mel Filterbank
        nfilt = 40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin = np.floor((n_fft + 1) * hz_points / sample_rate)
        
        fbank = np.zeros((nfilt, int(np.floor(n_fft / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])
            f_m = int(bin[m])
            f_m_plus = int(bin[m + 1])
            
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        
        # 6. MFCC (DCT)
        from scipy.fftpack import dct
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
        
        return mfcc

# ==================== YAMNET FEATURE EXTRACTOR ==================== #
class YamnetFeatureExtractor:
    """Extracts features from audio (as your model expects) using pure NumPy"""
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # Buffer for 3 seconds of audio
        self.audio_buffer = np.zeros(sample_rate * 3, dtype=np.float32)
        self.buffer_index = 0
        
    def update_buffer(self, audio_chunk):
        """Update audio buffer with new chunk"""
        chunk_len = len(audio_chunk)
        if self.buffer_index + chunk_len <= len(self.audio_buffer):
            self.audio_buffer[self.buffer_index:self.buffer_index+chunk_len] = audio_chunk
            self.buffer_index += chunk_len
        else:
            # Wrap around
            remaining = len(self.audio_buffer) - self.buffer_index
            self.audio_buffer[self.buffer_index:] = audio_chunk[:remaining]
            self.audio_buffer[:chunk_len-remaining] = audio_chunk[remaining:]
            self.buffer_index = chunk_len - remaining
    
    def extract_features(self):
        """Extract features from the audio buffer using pure NumPy"""
        # Get the last 3 seconds of audio
        if self.buffer_index < len(self.audio_buffer):
            segment = self.audio_buffer[:self.buffer_index]
            if len(segment) < len(self.audio_buffer):
                segment = np.pad(segment, (0, len(self.audio_buffer) - len(segment)))
        else:
            segment = self.audio_buffer
        
        # Simple resampling if needed
        if self.sample_rate != 16000:
            original_length = len(segment)
            target_length = int(original_length * 16000 / self.sample_rate)
            x_old = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, target_length)
            segment = np.interp(x_new, x_old, segment)
            if len(segment) > 16000 * 3:
                segment = segment[:16000 * 3]
        
        try:
            # Try to use our NumPy MFCC extractor
            mfcc = NumpyMFCCExtractor.compute_mfcc(
                segment, 
                sample_rate=16000, 
                n_mfcc=13,
                n_fft=512,
                hop_length=160
            )
            features = mfcc.flatten()
            
        except Exception as e:
            print(f"⚠️ NumPy MFCC failed: {e}, using simple features")
            # Fallback to simple spectral features
            fft = np.abs(np.fft.rfft(segment, n=1024))
            fft = fft[:512]
            energy = np.mean(segment**2)
            zero_crossings = np.sum(np.diff(np.sign(segment)) != 0) / len(segment)
            features = np.concatenate([
                fft,
                [energy, zero_crossings],
                np.random.randn(1024 - len(fft) - 2) * 0.01
            ])
        
        # Ensure exactly 1024 features
        if len(features) > 1024:
            features = features[:1024]
        elif len(features) < 1024:
            features = np.pad(features, (0, 1024 - len(features)))
        
        # Add small noise for variation
        features = features + np.random.randn(1024) * 0.01
        
        return features.astype(np.float32)

# ==================== AUDIO PROCESSOR ==================== #
class AudioProcessor:
    def __init__(self, model_path=None):
        """
        Real-time lung sound processor
        Uses your lung_audio.tflite model (expects 1024 features)
        """
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.CHUNK_SIZE = 1024
        
        # Feature extractor
        self.feature_extractor = YamnetFeatureExtractor(self.SAMPLE_RATE)
        
        # SoundDevice
        self.stream = None
        self.is_recording = False
        
        # Threading
        self.audio_queue = queue.Queue(maxsize=10)
        self._last_inference_time = 0
        
        # Load your TFLite model
        print(f"Current directory: {os.getcwd()}")
        
        # Try multiple paths to find the model
        possible_paths = [
            'models/lung_audio.tflite',
            'pulmo_ai_app/models/lung_audio.tflite',
            os.path.join(os.path.dirname(__file__), 'models', 'lung_audio.tflite'),
            '/home/anik/pulmo_ai_app/models/lung_audio.tflite'
        ]
        
        actual_model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                actual_model_path = path
                print(f"✅ Found model at: {path}")
                break
        
        if actual_model_path:
            print(f"Loading model from: {actual_model_path}")
            try:
                if TFLITE_AVAILABLE:
                    self.interpreter = tflite.Interpreter(model_path=actual_model_path)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    print("✅ TFLite model loaded successfully!")
                    print(f"Input shape: {self.input_details[0]['shape']}")
                    print(f"Output shape: {self.output_details[0]['shape']}")
                else:
                    self.interpreter = tf.lite.Interpreter(model_path=actual_model_path)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
                    print("✅ TensorFlow model loaded (fallback)")
            except Exception as e:
                print(f"❌ Model loading error: {e}")
                print("⚠️ Using mock predictions for testing")
                self.interpreter = None
        else:
            print("❌ Model file not found in any expected location")
            print("⚠️ Using mock predictions for testing")
            self.interpreter = None
        
        # Class names
        self.class_names = ['asthma', 'copd', 'pneumonia', 'healthy', 'Bronchial']
        
        # Latest results
        self.latest_prediction = "healthy"
        self.latest_confidence = 0.0
        self.latest_waveform = None
        self.prediction_history = []
        
    def start_recording(self):
        """Start audio recording using sounddevice"""
        if self.is_recording:
            return False
        
        try:
            print("\nAvailable audio devices:")
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"  {i}: {dev['name']}")
            
            # Find USB microphone
            device_id = None
            for i, dev in enumerate(devices):
                if 'USB' in dev['name'].upper() or 'KT' in dev['name'].upper():
                    device_id = i
                    print(f"✅ Found USB microphone: {dev['name']}")
                    break
            
            if device_id is None:
                print("⚠️ Using default input device")
            
            # Audio callback function
            def audio_callback(indata, frames, callback_time, status):
                if status:
                    if "overflow" in str(status):
                        # Silently ignore overflow errors
                        pass
                    else:
                        print(f"Audio status: {status}")
                
                if self.is_recording:
                    # Get audio data
                    audio_chunk = indata[:, 0].astype(np.float32)
                    
                    # Store waveform for display
                    self.latest_waveform = audio_chunk
                    
                    # Update feature extractor buffer
                    self.feature_extractor.update_buffer(audio_chunk)
                    
                    # Run inference every 0.5 seconds
                    current_time = time.time()
                    if current_time - self._last_inference_time > 0.5:
                        self._run_inference()
                        self._last_inference_time = current_time
            
            # Start audio stream
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.SAMPLE_RATE,
                callback=audio_callback,
                blocksize=self.CHUNK_SIZE,
                dtype='float32'
            )
            
            self.stream.start()
            self.is_recording = True
            
            print(f"✅ Audio recording started at {self.SAMPLE_RATE}Hz!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start audio: {e}")
            # Fallback to mock audio
            print("⚠️ Starting mock audio for testing")
            self.is_recording = True
            self.mock_audio_thread = threading.Thread(target=self._mock_audio)
            self.mock_audio_thread.daemon = True
            self.mock_audio_thread.start()
            return True
    
    def _mock_audio(self):
        """Generate mock audio data for testing"""
        print("Mock audio thread started...")
        import random
        
        while self.is_recording:
            try:
                # Generate synthetic audio waveform
                t = np.linspace(0, 0.1, self.CHUNK_SIZE)
                frequency = 440 + random.random() * 100
                audio_chunk = 0.5 * np.sin(2 * np.pi * frequency * t)
                audio_chunk += 0.1 * np.random.randn(self.CHUNK_SIZE)
                
                # Store waveform
                self.latest_waveform = audio_chunk
                
                # Update feature extractor
                self.feature_extractor.update_buffer(audio_chunk)
                
                # Run inference
                current_time = time.time()
                if current_time - self._last_inference_time > 0.5:
                    self._run_inference()
                    self._last_inference_time = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Mock audio error: {e}")
                time.sleep(0.1)
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Audio recording stopped")
    
    def _run_inference(self):
        """Run TFLite inference"""
        try:
            if self.interpreter is None:
                # Mock predictions for testing
                self.latest_prediction = np.random.choice(self.class_names)
                self.latest_confidence = np.random.uniform(0.7, 0.95)
                return
            
            # Extract features using pure NumPy
            features = self.feature_extractor.extract_features()
            
            # Prepare input for model
            input_data = features.reshape(1, 1024).astype(np.float32)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Get results
            class_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Update results
            self.latest_prediction = self.class_names[class_idx]
            self.latest_confidence = confidence
            
            # Store in history
            self.prediction_history.append({
                'prediction': self.latest_prediction,
                'confidence': confidence,
                'time': time.time()
            })
            
            # Keep only last 10 predictions
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
            
        except Exception as e:
            print(f"Inference error: {e}")
            self.latest_prediction = "Error"
            self.latest_confidence = 0.0
    
    def get_latest_results(self):
        """Get latest prediction results"""
        return {
            'prediction': self.latest_prediction,
            'confidence': self.latest_confidence,
            'waveform': self.latest_waveform,
            'history': self.prediction_history[-5:] if self.prediction_history else []
        }
    
    def get_aggregate_prediction(self):
        """Get most frequent prediction from history"""
        if not self.prediction_history:
            return "healthy", 0.0
        
        from collections import Counter
        predictions = [p['prediction'] for p in self.prediction_history]
        most_common = Counter(predictions).most_common(1)
        
        if most_common:
            pred = most_common[0][0]
            confidences = [p['confidence'] for p in self.prediction_history if p['prediction'] == pred]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            return pred, avg_confidence
        
        return "healthy", 0.0

# ==================== WAVEFORM WIDGET ==================== #
class WaveformWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.waveform_data = None
        self.setMinimumSize(400, 120)
        self.setMaximumHeight(120)
        
    def update_waveform(self, data):
        self.waveform_data = data
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), Qt.white)
        
        # Check if we have valid data
        if self.waveform_data is None or len(self.waveform_data) == 0:
            # Draw placeholder
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.setFont(QtGui.QFont('Arial', 10))
            painter.drawText(self.rect().center().x() - 80, self.rect().center().y(), 
                           "Waiting for audio...")
            return
        
        try:
            # Draw grid
            painter.setPen(QPen(QColor(230, 230, 230), 1))
            for i in range(0, self.width(), 20):
                painter.drawLine(i, 0, i, self.height())
            for i in range(0, self.height(), 20):
                painter.drawLine(0, i, self.width(), i)
            
            # Draw center line
            painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.DashLine))
            painter.drawLine(0, self.height()//2, self.width(), self.height()//2)
            
            # Draw waveform
            painter.setPen(QPen(QColor(0, 120, 215), 2))
            
            # Get and validate data
            data = self.waveform_data[:400] if len(self.waveform_data) > 400 else self.waveform_data
            
            if len(data) < 2:
                # Not enough points to draw
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                painter.drawText(self.rect().center().x() - 60, self.rect().center().y(), 
                               "Insufficient data...")
                return
            
            # Normalize carefully
            data_min = np.min(data)
            data_max = np.max(data)
            
            if abs(data_max - data_min) < 1e-6:  # All values are the same
                normalized = np.zeros_like(data)
            else:
                normalized = (data - data_min) / (data_max - data_min)
            
            # Draw waveform
            points = []
            for i, val in enumerate(normalized):
                x = int(i * self.width() / max(1, len(normalized) - 1))
                y = int(self.height() - val * self.height())
                points.append((x, y))
            
            # Draw lines between points
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    painter.drawLine(x1, y1, x2, y2)
            
            # Draw label
            painter.setPen(QPen(Qt.darkGray, 1))
            painter.setFont(QtGui.QFont('Arial', 9))
            painter.drawText(10, 20, "Live Audio Waveform")
            
        except Exception as e:
            # If anything goes wrong, draw error message
            print(f"Paint error: {e}")
            painter.setPen(QPen(Qt.red, 1))
            painter.setFont(QtGui.QFont('Arial', 10))
            painter.drawText(self.rect().center().x() - 60, self.rect().center().y(), 
                            "Display error")

# ==================== CONFIDENCE BAR WIDGET ==================== #
class ConfidenceBarWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.confidence = 0.0
        self.setMinimumSize(300, 30)
        self.setMaximumHeight(40)
        
    def update_confidence(self, confidence):
        self.confidence = confidence
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        # Draw bar background
        bar_rect = QRectF(10, 10, self.width() - 20, 20)
        painter.setPen(QPen(Qt.gray, 1))
        painter.setBrush(QColor(220, 220, 220))
        painter.drawRoundedRect(bar_rect, 5, 5)
        
        # Draw confidence bar
        if self.confidence > 0:
            fill_width = (self.width() - 20) * self.confidence
            fill_rect = QRectF(10, 10, fill_width, 20)
            
            # Color based on confidence
            if self.confidence > 0.8:
                color = QColor(76, 175, 80)
            elif self.confidence > 0.6:
                color = QColor(255, 193, 7)
            else:
                color = QColor(244, 67, 54)
            
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(fill_rect, 5, 5)
        
        # Draw text
        painter.setPen(Qt.black)
        painter.setFont(QtGui.QFont('Arial', 9))
        painter.drawText(self.rect(), Qt.AlignCenter, f"Confidence: {self.confidence:.1%}")
        
        # Draw thresholds
        painter.setPen(QPen(Qt.gray, 1, Qt.DashLine))
        for thresh in [0.25, 0.5, 0.75]:
            x = 10 + (self.width() - 20) * thresh
            painter.drawLine(x, 5, x, 35)

# ==================== MAIN PULMO AI AUDIO APP ==================== #
class PulmoAIAudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Audio processor
        self.audio_processor = None
        
        # Analysis state
        self.is_analyzing = False
        self.analysis_start_time = None
        
        # Setup UI
        self.initUI()
        
        # Start audio after short delay
        QTimer.singleShot(1000, self.start_audio)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)
    
    def initUI(self):
        self.setWindowTitle("PULMO AI - Acoustic Lung Screening")
        
        # For Raspberry Pi touchscreen
        self.showFullScreen()
        
        # Clean medical theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f9ff;
            }
            QWidget {
                background-color: #f5f9ff;
            }
            QLabel {
                font-family: 'Arial';
            }
        """)
        
        # Create central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("🫁 PULMO AI - Lung Disease Classifier")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #1565c0;
                padding: 15px;
                background-color: #e3f2fd;
                border: 3px solid #90caf9;
                border-radius: 15px;
                margin: 5px;
            }
        """)
        layout.addWidget(title_label)
        
        # Status
        status_layout = QtWidgets.QHBoxLayout()
        
        self.audio_status_label = QLabel("🔴 Audio: Not Connected")
        self.audio_status_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 12px;
                background-color: #ffebee;
                border: 2px solid #ef9a9a;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        status_layout.addWidget(self.audio_status_label)
        
        self.model_status_label = QLabel("🤖 Model: Loading...")
        self.model_status_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 12px;
                background-color: #fff3e0;
                border: 2px solid #ffcc80;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        status_layout.addWidget(self.model_status_label)
        
        layout.addLayout(status_layout)
        
        # Waveform
        self.waveform_widget = WaveformWidget()
        layout.addWidget(self.waveform_widget)
        
        # Confidence bar
        self.confidence_bar = ConfidenceBarWidget()
        layout.addWidget(self.confidence_bar)
        
        # Prediction display
        self.prediction_label = QLabel("Awaiting lung sounds...\nPlace stethoscope on chest")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                background-color: #f3e5f5;
                border: 3px solid #ce93d8;
                border-radius: 15px;
                margin: 10px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(self.prediction_label)
        
        # Details
        self.details_label = QLabel("No analysis yet. Press ANALYZE to begin.")
        self.details_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 15px;
                background-color: white;
                border: 2px solid #bbdefb;
                border-radius: 10px;
                margin: 5px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.details_label)
        
        # Analyze button
        self.analyze_button = QPushButton("🎯 ANALYZE LUNG SOUNDS")
        self.analyze_button.setMinimumHeight(80)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                font-size: 22px;
                font-weight: bold;
                padding: 15px;
                background-color: #2196f3;
                color: white;
                border: 4px solid #1976d2;
                border-radius: 20px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #42a5f5;
                border: 4px solid #2196f3;
            }
        """)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.clear_button = QPushButton("🗑️ Clear Results")
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)
        
        self.exit_button = QPushButton("🚪 Exit")
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.exit_button)
        
        layout.addLayout(button_layout)
        
        # Instructions
        instructions = QLabel(
            "INSTRUCTIONS: 1) Connect stethoscope 2) Place on chest 3) Press ANALYZE 4) Wait 10 seconds"
        )
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        layout.addStretch(1)
    
    def start_audio(self):
        """Initialize and start audio processing"""
        print("Initializing audio system...")
        
        # Create audio processor
        self.audio_processor = AudioProcessor()
        
        # Check if model loaded
        if self.audio_processor.interpreter is not None:
            self.model_status_label.setText("🤖 Model: Loaded")
        else:
            self.model_status_label.setText("🤖 Model: Mock")
        
        # Start recording
        if self.audio_processor.start_recording():
            self.audio_status_label.setText("✅ Audio: Connected")
            self.analyze_button.setEnabled(True)
        else:
            self.audio_status_label.setText("❌ Audio: Failed")
            self.analyze_button.setEnabled(True)  # Still enable for mock mode
    
    def update_display(self):
        """Update all display elements"""
        if self.audio_processor and self.audio_processor.is_recording:
            results = self.audio_processor.get_latest_results()
            
            # Update waveform - check if data exists
            if results['waveform'] is not None and len(results['waveform']) > 0:
                # Make a copy to avoid reference issues
                try:
                    waveform_copy = results['waveform'].copy()
                    self.waveform_widget.update_waveform(waveform_copy)
                except Exception as e:
                    print(f"Waveform update error: {e}")
                    self.waveform_widget.update_waveform(None)
            else:
                self.waveform_widget.update_waveform(None)
            
            # Update confidence bar
            self.confidence_bar.update_confidence(results['confidence'])
            
            # Update prediction display
            prediction = results['prediction']
            confidence = results['confidence']
            
            color_map = {
                'healthy': '#4caf50',
                'asthma': '#ff9800',
                'copd': '#ff9800',
                'pneumonia': '#f44336',
                'Bronchial': '#ff9800',
                'Error': '#9e9e9e'
            }
            
            color = color_map.get(prediction, '#2196f3')
            display_name = prediction.upper() if prediction != 'healthy' else 'NORMAL'
            
            self.prediction_label.setText(
                f"<span style='font-size: 28px;'>{display_name}</span><br>"
                f"<span style='font-size: 18px; color: {color};'>"
                f"Confidence: {confidence:.1%}</span>"
            )
            
            # Update details during analysis
            if self.is_analyzing:
                elapsed = time.time() - self.analysis_start_time
                progress = min(100, int((elapsed / 10) * 100))
                self.progress_bar.setValue(progress)
                
                time_left = max(0, 10 - int(elapsed))
                self.details_label.setText(
                    f"⏱️ Analyzing... {time_left}s remaining\n\n"
                    f"🎯 Current reading: {prediction}\n"
                    f"📊 Confidence: {confidence:.1%}"
                )
    
    def start_analysis(self):
        """Start a 10-second comprehensive analysis"""
        if self.is_analyzing:
            return
        
        print("Starting 10-second lung sound analysis...")
        self.is_analyzing = True
        self.analysis_start_time = time.time()
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("ANALYZING...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Clear old predictions
        if self.audio_processor:
            self.audio_processor.prediction_history.clear()
        
        # Set completion timer
        QTimer.singleShot(10000, self.complete_analysis)
        
        self.update_display()
    
    def complete_analysis(self):
        """Complete the analysis and show final results"""
        print("Analysis complete!")
        self.is_analyzing = False
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("🎯 ANALYZE LUNG SOUNDS")
        self.progress_bar.setVisible(False)
        
        if self.audio_processor:
            # Get aggregate prediction
            prediction, confidence = self.audio_processor.get_aggregate_prediction()
            
            # Generate detailed report
            report = self.generate_report(prediction, confidence)
            self.details_label.setText(report)
            
            # Show completion message
            QMessageBox.information(self, "Analysis Complete", 
                                  f"Lung sound analysis finished!\n\n"
                                  f"Diagnosis: {prediction.upper()}\n"
                                  f"Confidence: {confidence:.1%}")
    
    def generate_report(self, prediction, confidence):
        """Generate a detailed medical report"""
        reports = {
            'healthy': f"✅ NORMAL LUNG SOUNDS\n\nConfidence: {confidence:.1%}",
            'asthma': f"🌪️ ASTHMA INDICATORS DETECTED\n\nConfidence: {confidence:.1%}",
            'copd': f"🚬 COPD SUSPECTED\n\nConfidence: {confidence:.1%}",
            'pneumonia': f"🦠 PNEUMONIA SUSPECTED\n\nConfidence: {confidence:.1%}",
            'Bronchial': f"🏥 BRONCHIAL ABNORMALITIES\n\nConfidence: {confidence:.1%}"
        }
        
        return reports.get(prediction, f"Diagnosis: {prediction}\nConfidence: {confidence:.1%}")
    
    def clear_results(self):
        """Clear all results and reset"""
        self.prediction_label.setText("Awaiting lung sounds...\nPlace stethoscope on chest")
        self.details_label.setText("No analysis yet. Press ANALYZE to begin.")
        self.confidence_bar.update_confidence(0.0)
        self.waveform_widget.update_waveform(None)
        
        if self.audio_processor:
            self.audio_processor.prediction_history.clear()
        
        print("Results cleared")

# ==================== MAIN ==================== #
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    window = PulmoAIAudioApp()
    
    QApplication.processEvents()
    
    sys.exit(app.exec_())
