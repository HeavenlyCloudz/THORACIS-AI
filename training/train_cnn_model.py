# train_cnn_model.py - UPDATED for 2 paths
"""
Training CNN on S21 "images" (2 paths × 201 frequencies)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_cnn_model(input_shape=(2, 201, 1), num_classes=3):
    """Create CNN for S21 image classification (2 paths)"""
    
    model = keras.Sequential([
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
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_dataset(dataset_path):
    """Load the dataset from your ml_dataset_final folder"""
    
    class_names = ['baseline', 'healthy', 'tumor']
    images = []
    labels = []
    
    # Find all numpy image files
    cnn_folder = dataset_path / '02_cnn_images'
    for npy_file in cnn_folder.glob('*.npy'):
        if 'image' in str(npy_file) and 'augmented' not in str(npy_file):
            # Determine class from filename
            filename = npy_file.stem
            if 'baseline' in filename:
                label = 0
            elif 'healthy' in filename:
                label = 1
            elif 'tumor' in filename:
                label = 2
            else:
                continue
            
            img = np.load(npy_file)
            # Add channel dimension (2, 201) -> (2, 201, 1)
            img = img.reshape(2, 201, 1)
            images.append(img)
            labels.append(label)
    
    # Load augmented images
    aug_folder = dataset_path / '03_cnn_augmented'
    if aug_folder.exists():
        for npy_file in aug_folder.glob('*.npy'):
            filename = npy_file.stem
            if 'tumor' in filename:
                img = np.load(npy_file)
                img = img.reshape(2, 201, 1)
                images.append(img)
                labels.append(2)  # Augmented tumor samples
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"✅ Loaded {len(images)} images")
    for i, name in enumerate(class_names):
        count = np.sum(labels == i)
        print(f"   {name}: {count} ({count/len(images)*100:.1f}%)")
    
    return images, labels, class_names

def main():
    print("="*70)
    print("PULMO AI: Train CNN on S21 Images")
    print("="*70)
    
    # Find latest dataset
    dataset_folders = sorted(Path('.').glob('ml_dataset_final_*'))
    if not dataset_folders:
        print("❌ No ml_dataset_final_* folders found!")
        return
    
    latest_dataset = dataset_folders[-1]
    print(f"\n📁 Using dataset: {latest_dataset}")
    
    # Load data
    images, labels, class_names = load_dataset(latest_dataset)
    
    if len(images) == 0:
        print("❌ No images found!")
        return
    
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
            str(latest_dataset / 'best_cnn_model.h5'), 
            save_best_only=True
        )
    ]
    
    # Train
    print("\n🏋️ Training CNN...")
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
    plt.savefig(latest_dataset / 'cnn_training_history.png', dpi=150)
    print(f"\n📊 Saved: cnn_training_history.png")
    
    # Evaluate
    val_loss, val_acc = model.evaluate(x_val, y_val)
    print(f"\n✅ Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model.save(latest_dataset / 'pulmo_cnn_model.h5')
    
    # Convert to TFLite for Raspberry Pi
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(latest_dataset / 'pulmo_cnn_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ Model saved to: {latest_dataset}/pulmo_cnn_model.h5")
    print(f"✅ TFLite model saved to: {latest_dataset}/pulmo_cnn_model.tflite")
    
    # Save class names
    with open(latest_dataset / 'class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    print("\n🚀 DONE! CNN model trained successfully!")

if __name__ == "__main__":
    main()