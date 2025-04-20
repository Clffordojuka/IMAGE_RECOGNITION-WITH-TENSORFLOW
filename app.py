from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the trained model
with keras.saving.custom_object_scope({'BatchNormalization': keras.layers.BatchNormalization}):
    model = load_model("model.keras")


# Set image size expected by the model
IMG_SIZE = (180, 180)
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

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