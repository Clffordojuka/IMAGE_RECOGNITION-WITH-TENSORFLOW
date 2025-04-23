# flower_classification.py

# === 1. Import Libraries ===
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
import tarfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report

# === 2. Download & Extract Dataset ===
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

# Remove LICENSE.txt if exists
for item in data_dir.iterdir():
    if item.is_file() and item.name == "LICENSE.txt":
        os.remove(item)

categories = [item.name for item in data_dir.iterdir() if item.is_dir()]
print("Categories (flowers):", categories)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Total Images:", image_count)

# === 3. Visualize Sample Images ===
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

# === 4. Load & Preprocess Dataset ===
image_size = (224, 224)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

print(f"Train dataset size: {len(train_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")

# === 5. Visualize Image Split ===
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# === 6. Data Augmentation & Prefetching ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === 7. Define Model using Transfer Learning ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 8. Train Top Layers ===
initial_epochs = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=initial_epochs)

# === 9. Fine-tune Base Model ===
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs)

# === 10. Evaluate & Visualize Performance ===
acc = history_finetune.history['accuracy']
val_acc = history_finetune.history['val_accuracy']
loss = history_finetune.history['loss']
val_loss = history_finetune.history['val_loss']
epochs_range = range(total_epochs)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()

# === 11. Classification Report ===
y_pred = np.concatenate([np.argmax(model.predict(x), axis=1) for x, _ in val_ds])
y_true = np.concatenate([y.numpy() for _, y in val_ds])
print(classification_report(y_true, y_pred, target_names=class_names))

# === 12. Save Model ===
model.save("mobilenetv2.h5")