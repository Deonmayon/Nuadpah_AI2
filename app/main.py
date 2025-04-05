from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os
import requests

app = FastAPI()

model = YOLO("model/new_best_seg.pt")

def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(url: str):
    try:
        # Get image from URL
        img = get_image(url)

        # Convert to numpy (YOLO expects ndarray or path)
        img_np = np.array(img)

        # Predict using YOLO
        results = model(img_np)

        # Extract result data
        result_data = []
        for result in results:
            boxes = result.boxes
            masks = result.masks
            for i in range(len(boxes)):
                box = boxes[i].xyxy.cpu().numpy().tolist()[0] if boxes is not None else None
                conf = boxes[i].conf.item() if boxes is not None else None
                cls = int(boxes[i].cls.item()) if boxes is not None else None
                class_name = model.names[cls] if cls is not None else None

                result_data.append({
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": box,
                })

        return {
            "success": True,
            "predictions": result_data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
        
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        # Read file and convert to NumPy array
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        # Predict using YOLO
        results = model(img_np)

        # Extract result data
        result_data = []
        for result in results:
            boxes = result.boxes
            masks = result.masks
            for i in range(len(boxes)):
                box = boxes[i].xyxy.cpu().numpy().tolist()[0] if boxes is not None else None
                conf = boxes[i].conf.item() if boxes is not None else None
                cls = int(boxes[i].cls.item()) if boxes is not None else None
                class_name = model.names[cls] if cls is not None else None

                result_data.append({
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": box,
                })

        return {
            "success": True,
            "predictions": result_data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }