from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.models import Sequential, load_model
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# Patch axis deserialization for legacy models
class PatchedBatchNorm(BatchNormalization):
    def __init__(self, axis=-1, **kwargs):
        if isinstance(axis, list):  # Convert lists to integers
            axis = axis[0]
        super().__init__(axis=int(axis), **kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = int(self.axis)  # Ensure axis is an integer
        return config

    @classmethod
    def from_config(cls, config):
        axis = config.get('axis', -1)
        if isinstance(axis, list):
            axis = axis[0]
        config['axis'] = int(axis)
        return cls(**config)

def create_model(num_classes): # Added num_classes as parameter
  model = Sequential([
      Input(shape=(180, 180, 3)), # Input layer instead
      layers.Rescaling(1./255),  # Normalize pixel values to [0,1]

      layers.Conv2D(16, 3, padding='same', activation='relu'),  # 1st Conv layer
      BatchNormalization(axis=3), # Batch Norm after Conv2D - PATCH HERE
      layers.MaxPooling2D(),  # Pooling reduces feature map size

      layers.Conv2D(32, 3, padding='same', activation='relu'),  # 2nd Conv layer
      BatchNormalization(axis=3), # Batch Norm after Conv2D - PATCH HERE
      layers.MaxPooling2D(),

      layers.Conv2D(64, 3, padding='same', activation='relu'),  # 3rd Conv layer
      BatchNormalization(axis=3), # Batch Norm after Conv2D - PATCH HERE
      layers.MaxPooling2D(),

      layers.Flatten(),  # Convert 2D feature maps into a 1D vector
      layers.Dense(128, activation='relu'),  # Fully connected layer
      layers.Dense(num_classes, activation='softmax')  # Output layer with `num_classes` neurons
  ])
  return model

# Number of output classes
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
num_classes = len(CLASS_NAMES)  # Number of output classes

# Get the model
model = create_model(num_classes) # Passed num_classes

# Compile the model - VERY IMPORTANT
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Set to False
              metrics=['accuracy'])

# Save the entire model in .h5 format.
model.save("model.h5")

# Load model with custom handler
model = load_model("model.h5", custom_objects={'BatchNormalization': PatchedBatchNorm})

# Set image size expected by the model
IMG_SIZE = (180, 180)

@app.get("/")
def read_root():
    return {"message": "🌸 Flower Classifier API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        return {
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)