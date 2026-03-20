# train_cnn_model.py
"""
Training my CNN on my S21 "images"
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_cnn_model(input_shape=(4, 201, 1), num_classes=3):
    """Create CNN for S21 image classification"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (2, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (2, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (2, 3), activation='relu', padding='same'),
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

def load_dataset(dataset_path):
    """Load the augmented dataset"""
    
    class_names = ['baseline', 'healthy', 'tumor']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    images = []
    labels = []
    metadata = []
    
    # Load original numpy files
    for class_name in class_names:
        numpy_path = Path(dataset_path) / class_name / 'numpy'
        for npy_file in numpy_path.glob('*.npy'):
            if '_meta' not in str(npy_file):
                img = np.load(npy_file)
                # Add channel dimension
                img = img.reshape(4, 201, 1)
                images.append(img)
                labels.append(class_to_idx[class_name])
                
                # Load metadata if exists
                meta_file = npy_file.with_name(npy_file.stem + '_meta.json')
                if meta_file.exists():
                    with open(meta_file) as f:
                        metadata.append(json.load(f))
    
    # Load augmented files
    for class_name in class_names:
        aug_path = Path(dataset_path) / class_name / 'augmented'
        for npy_file in aug_path.glob('*.npy'):
            img = np.load(npy_file)
            img = img.reshape(4, 201, 1)
            images.append(img)
            labels.append(class_to_idx[class_name])
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"✅ Loaded {len(images)} images")
    for i, name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"   {name}: {count} ({count/len(images)*100:.1f}%)")
    
    return images, labels, class_names

def main():
    print("=== PULMO AI: Train CNN on S21 Images ===\n")
    
    # Find latest dataset
    dataset_folders = sorted(Path('.').glob('ml_dataset_*'))
    if not dataset_folders:
        print("❌ No ml_dataset_* folders found!")
        return
    
    latest_dataset = dataset_folders[-1]
    print(f"📁 Using dataset: {latest_dataset}")
    
    # Load data
    images, labels, class_names = load_dataset(latest_dataset)
    
    # Shuffle and split
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    split = int(0.8 * len(images))
    
    x_train, x_val = images[:split], images[split:]
    y_train, y_val = labels[:split], labels[split:]
    
    print(f"\n📊 Train: {len(x_train)}, Validation: {len(x_val)}")
    
    # Create model
    model = create_cnn_model()
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
            f'{latest_dataset}/best_model.h5', 
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
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
    
    # Save model in multiple formats
    model.save(f'{latest_dataset}/pulmo_cnn_model.h5')
    
    # Convert to TFLite for Raspberry Pi
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(f'{latest_dataset}/pulmo_cnn_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Model saved as:")
    print(f"   - {latest_dataset}/pulmo_cnn_model.h5")
    print(f"   - {latest_dataset}/pulmo_cnn_model.tflite (for Raspberry Pi)")
    
    # Save class names
    with open(f'{latest_dataset}/class_names.json', 'w') as f:
        json.dump(class_names, f)

if __name__ == "__main__":
    main()
