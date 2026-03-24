# train_cnn_model.py - UPDATED with .keras format
"""
Training CNN on S21 "images" from PULMO AI
Saves model as .keras file for Raspberry Pi deployment
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

def create_cnn_model(input_shape=(2, 201, 1), num_classes=3):
    """Create CNN for S21 image classification (2 paths only)"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (1, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (1, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (1, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    print("="*60)
    print("PULMO AI: Train CNN on S21 Images")
    print("="*60)
    
    # Find latest dataset
    dataset_folders = sorted(Path('.').glob('ml_dataset_final_*'))
    if not dataset_folders:
        print("❌ No ml_dataset_final_* folders found!")
        return
    
    latest_dataset = dataset_folders[-1]
    print(f"\n📁 Using dataset: {latest_dataset}")
    
    # Load CNN images
    cnn_images_path = latest_dataset / '02_cnn_images' / 'cnn_images.npy'
    cnn_labels_path = latest_dataset / '02_cnn_images' / 'cnn_labels.npy'
    
    if not cnn_images_path.exists():
        print("❌ CNN images not found! Run create_ml_images_final.py first")
        return
    
    images = np.load(cnn_images_path)
    labels = np.load(cnn_labels_path)
    
    print(f"✅ Loaded {len(images)} images")
    print(f"   Image shape: {images.shape}")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # Shuffle and split
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    split = int(0.7 * len(images))
    x_train, x_val = images[:split], images[split:]
    y_train, y_val = labels[:split], labels[split:]
    
    print(f"\n📊 Train: {len(x_train)}, Validation: {len(x_val)}")
    
    # Create model
    model = create_cnn_model(input_shape=(2, 201, 1), num_classes=3)
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            f'{latest_dataset}/best_model.keras', 
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{latest_dataset}/training_history.png', dpi=150)
    print(f"\n📊 Saved training history plot")
    
    # Evaluate
    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"\n✅ Validation Accuracy: {val_acc:.4f}")
    
    # Save model as .keras (preferred format)
    model.save(f'{latest_dataset}/pulmo_cnn_model.keras')
    print(f"✅ Model saved: {latest_dataset}/pulmo_cnn_model.keras")
    
    # Convert to TFLite for Raspberry Pi
    print("\n🔄 Converting to TFLite for Raspberry Pi...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(f'{latest_dataset}/pulmo_cnn_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved: {latest_dataset}/pulmo_cnn_model.tflite")
    
    # Save class names
    class_names = ['baseline', 'healthy', 'tumor']
    with open(f'{latest_dataset}/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    print("\n🚀 Done! Model ready for Raspberry Pi deployment")

if __name__ == "__main__":
    main()
