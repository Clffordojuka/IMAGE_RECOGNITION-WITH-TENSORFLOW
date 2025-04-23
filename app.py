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

# Custom object to handle BatchNormalization layer deserialization
class PatchedBatchNorm(BatchNormalization):
    def __init__(self, axis=-1, **kwargs):
        if isinstance(axis, list):
            axis = axis[0]
        super().__init__(axis=int(axis), **kwargs)

    @classmethod
    def from_config(cls, config):
        axis = config.get('axis')
        if isinstance(axis, list):
            axis = axis[0]
        config['axis'] = int(axis)
        return cls(**config)

def create_model(num_classes):
  model = Sequential([
      Input(shape=(180, 180, 3)),
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      BatchNormalization(axis=3),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      BatchNormalization(axis=3),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      BatchNormalization(axis=3),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes, activation='softmax')
  ])
  return model

CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
num_classes = len(CLASS_NAMES)

model = create_model(num_classes)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.save("model.h5")

# Load model with the custom object
model = load_model("model.h5", custom_objects={'BatchNormalization': PatchedBatchNorm})

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