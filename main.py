from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("model.h5")

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    processed = preprocess(img)
    pred = model.predict(processed)[0][0]
    
    return {
        "verdict": "FAKE" if pred > 0.5 else "REAL",
        "confidence": float(pred if pred > 0.5 else 1-pred)
    }