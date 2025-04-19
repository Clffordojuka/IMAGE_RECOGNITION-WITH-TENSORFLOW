#!/usr/bin/env python
# coding: utf-8

# ========================================
#           IMPORT LIBRARIES
# ========================================

import os
import pathlib
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report


# ========================================
#           LOAD AND EXTRACT DATASET
# ========================================

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
download_path = tf.keras.utils.get_file("flower_photos.tgz", origin=dataset_url, cache_dir=".", extract=False)
extract_folder = "./flower_photos"

if not os.path.exists(extract_folder):
    with tarfile.open(download_path, 'r:gz') as tar:
        tar.extractall(path=".")
    print("Dataset extracted successfully.")
else:
    print("Dataset already extracted.")

data_dir = pathlib.Path(extract_folder)
if data_dir.exists() and any(data_dir.iterdir()):
    categories = [item.name for item in data_dir.iterdir() if item.is_dir()]
    print("Categories (flowers):", categories)
else:
    print("Dataset extraction failed or directory is empty.")


# ========================================
#           CLEAN UP AND VISUALIZE
# ========================================

# Remove LICENSE.txt if exists
license_file = data_dir / "LICENSE.txt"
if license_file.exists():
    print("Removing LICENSE.txt...")
    os.remove(license_file)

categories = [item.name for item in data_dir.iterdir() if item.is_dir()]
print("Remaining categories:", categories)

image_count = len(list(data_dir.glob('*/*.jpg')))
print("Total image count:", image_count)

# Plot one sample image per category
flower_categories = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, category in enumerate(flower_categories):
    image_path = list(data_dir.glob(f'{category}/*'))[0]
    img = PIL.Image.open(str(image_path))
    axs[i].imshow(img)
    axs[i].set_title(category.capitalize())
    axs[i].axis('off')

plt.tight_layout()
plt.show()


# ========================================
#           TRAIN-TEST SPLIT
# ========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

print(f"Train batches: {len(train_ds)}, Validation batches: {len(val_ds)}")
class_names = train_ds.class_names
print("Classes:", class_names)


# ========================================
#           DISPLAY SAMPLE TRAIN IMAGES
# ========================================

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# ========================================
#           BASELINE MODEL
# ========================================

num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# ========================================
#           TRAIN BASELINE MODEL
# ========================================

initial_epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=initial_epochs)


# ========================================
#           PLOT TRAINING METRICS
# ========================================

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(initial_epochs)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()


# ========================================
#           DATA AUGMENTATION
# ========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])


# ========================================
#           TRANSFER LEARNING MODEL
# ========================================

base_model = tf.keras.applications.MobileNetV2(input_shape=(180, 180, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.build(input_shape=(None, 180, 180, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ========================================
#           TRAIN & FINE-TUNE MODEL
# ========================================

# Step 1: Train only top layers
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Step 2: Fine-tune base model
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 3: Fine-tune entire model
history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=10)


# ========================================
#           EVALUATE FINE-TUNED MODEL
# ========================================

acc = history_finetune.history['accuracy']
val_acc = history_finetune.history['val_accuracy']
loss = history_finetune.history['loss']
val_loss = history_finetune.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc, label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()


# ========================================
#           CLASSIFICATION REPORT
# ========================================

y_pred = np.concatenate([np.argmax(model.predict(x), axis=1) for x, _ in val_ds])
y_true = np.concatenate([y.numpy() for _, y in val_ds])

print(classification_report(y_true, y_pred, target_names=class_names))


# ========================================
#           SAVE FINAL MODEL
# ========================================

model.save('fine_tuned_flower_model.h5')