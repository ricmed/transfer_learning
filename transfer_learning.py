# -*- coding: utf-8 -*-
"""transfer_learning.py

Adapted for Cats and Dogs dataset with Transfer Learning (VGG16).
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt

# --- Configuration ---
# Set the number of images to use per category (randomly selected)
# Adjust this number based on your needs and hardware capabilities.
NUM_IMAGES_PER_CLASS = 1000 
BATCH_SIZE = 32
EPOCHS = 10
ROOT_DIR = 'PetImages'
CLASSES = ['Cat', 'Dog']

# --- GPU Configuration ---
# Check for GPU and configure memory growth for MX150 or similar GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print("GPU is available and configured.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("GPU not found. Using CPU.")

# --- Data Loading ---

def get_image(path):
    """Load and preprocess an image."""
    try:
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None, None

def load_data(root_dir, classes, num_samples_per_class):
    """Load a random subset of images from the dataset."""
    data = []
    print(f"Loading data from {root_dir}...")
    
    for c, category in enumerate(classes):
        class_dir = os.path.join(root_dir, category)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue
            
        all_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly select n images
        if len(all_images) > num_samples_per_class:
            selected_images = random.sample(all_images, num_samples_per_class)
        else:
            selected_images = all_images
            print(f"Warning: Only {len(all_images)} images found for {category}, using all.")

        print(f"Loading {len(selected_images)} images for category: {category}")
        
        for img_path in selected_images:
            _, x = get_image(img_path)
            if x is not None:
                data.append({'x': np.array(x[0]), 'y': c})
    
    return data

# Load data
data = load_data(ROOT_DIR, CLASSES, NUM_IMAGES_PER_CLASS)

if not data:
    print("No data loaded. Exiting.")
    exit()

# Shuffle data
random.shuffle(data)

# Split data (70% train, 15% val, 15% test)
train_split, val_split = 0.7, 0.15
idx_val = int(train_split * len(data))
idx_test = int((train_split + val_split) * len(data))

train = data[:idx_val]
val = data[idx_val:idx_test]
test = data[idx_test:]

# Prepare arrays
x_train = np.array([t["x"] for t in train])
y_train = np.array([t["y"] for t in train])
x_val = np.array([t["x"] for t in val])
y_val = np.array([t["y"] for t in val])
x_test = np.array([t["x"] for t in test])
y_test = np.array([t["y"] for t in test])

# Normalize (VGG16 preprocess_input usually handles this, but if we want 0-1 range for other models)
# However, VGG16 preprocess_input centers the data. 
# The original script divided by 255. Let's stick to standard VGG preprocessing which we already did in get_image.
# BUT, the original script did `x_train = x_train.astype('float32') / 255.` AFTER `preprocess_input`.
# This is actually double preprocessing or potentially incorrect if using `keras.applications.vgg16.preprocess_input` which expects 0-255 range BGR.
# Let's trust `preprocess_input` from keras and NOT divide by 255 again if we want to use the pretrained weights correctly.
# Wait, the original script used `imagenet_utils.preprocess_input`.
# Let's verify: Keras VGG16 preprocess_input takes RGB 0-255 and converts to BGR centered.
# If we divide by 255, we break it.
# I will REMOVE the division by 255 to be correct for VGG16 transfer learning.

# Convert labels to one-hot
num_classes = len(CLASSES)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"Data loaded: {len(data)} images.")
print(f"Train: {x_train.shape}, {y_train.shape}")
print(f"Val: {x_val.shape}, {y_val.shape}")
print(f"Test: {x_test.shape}, {y_test.shape}")


# --- Model Setup (Transfer Learning) ---

print("Loading VGG16 model...")
# Load VGG16 without the top layers
vgg = keras.applications.VGG16(weights='imagenet', include_top=True)
# vgg.summary()

# Create new model
inp = vgg.input
# We want to replace the last layer. VGG16 last layer is predictions (1000).
# Layer before that is 'fc2' (4096).
# We can just take the output of the second to last layer.
new_classification_layer = Dense(num_classes, activation='softmax', name='predictions_custom')(vgg.layers[-2].output)
model_new = Model(inp, new_classification_layer)

# Freeze all layers except the last one
for layer in model_new.layers[:-1]:
    layer.trainable = False

model_new.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model_new.summary()

# --- Training ---

print("Starting training...")
history = model_new.fit(x_train, y_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS,
                         validation_data=(x_val, y_val))

# --- Evaluation ---

print("Evaluating on test set...")
loss, accuracy = model_new.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# --- Save Model ---
model_new.save('cats_dogs_transfer_model.h5')
print("Model saved to cats_dogs_transfer_model.h5")

# --- Plotting (Optional) ---
# Save plot to file instead of showing
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label='Train Loss')
plt.plot(history.history["val_loss"], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label='Train Acc') # 'accuracy' in newer keras, 'acc' in older
plt.plot(history.history["val_accuracy"], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.savefig('training_history.png')
print("Training history saved to training_history.png")