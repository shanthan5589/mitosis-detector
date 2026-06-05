from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

from inference import predict_tile

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Mitosis Detector API'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {'error': 'Could not read image'}

    result = predict_tile(img_bgr)
    return result