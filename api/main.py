from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from src.model import load_model
from src.predict import predict, predict_from_landmarks
from monitoring.metrics import (
    start_metrics_collection, 
    INFERENCE_TIME_HISTOGRAM, 
    MODEL_PREDICTION_TYPES, 
    INVALID_LANDMARK_ERRORS, 
    REQUEST_COUNT, 
    REQUEST_ERRORS, 
    REQUEST_LATENCY
)        
# from threading import Thread
import logging
import numpy as np
import pandas as pd
import cv2
import asyncio
import mediapipe as mp
from typing import List
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model, le, detector = load_model()
Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
def startup_event():
    start_metrics_collection()
    # start_continuous_prediction(model, le, detector)
    
@app.get("/")
def home():
    logger.info("Home endpoint hit")
    return {"message": "Welcome to the Hand Gesture API"}

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    return """
    <html>
    <head>
        <title>Hand Gesture Recognition</title>
    </head>
    <body>
        <h2>Gesture Prediction</h2>
        <label for="delay">Delay (seconds between predictions):</label>
        <input type="number" id="delay" value="0" min="0" step="0.1">
        <br><br>
        <video id="video" width="640" height="480" autoplay></video><br/>
        <button onclick="capture()">Predict</button>
        <p id="result"></p>
        <script>
            const video = document.getElementById('video');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                });

            function capture() {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);

                canvas.toBlob(blob => {
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');

                    const delay = document.getElementById("delay").value;

                    fetch(`/predict-image?delay=${delay}`, {
                        method: 'POST',
                        body: formData
                    })
                    .then(res => res.json())
                    .then(data => {
                        document.getElementById('result').innerText = "Gesture: " + data.gesture;
                    });
                }, 'image/jpeg');
            }
        </script>
    </body>
    </html>
    """



@app.get("/continue-prediction", response_class=HTMLResponse)
async def continue_prediction_ui():
    return """
    <html>
    <head><title>Continuous Gesture Prediction</title></head>
    <body>
        <h2>Continuous Gesture Prediction</h2>
        <p>Prediction: <span id="prediction">Waiting...</span></p>
        <video id="video" autoplay playsinline style="display:none;"></video>
        <script>
        const video = document.getElementById('video');
        const predictionSpan = document.getElementById('prediction');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { predictionSpan.textContent = 'Camera access denied'; });

        async function sendFrame() {
            if (!video.videoWidth) {
                setTimeout(sendFrame, 500);
                return;
            }

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));

            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const response = await fetch('/predict-image', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                predictionSpan.textContent = data.gesture || 'No hand detected';
            } catch (e) {
                predictionSpan.textContent = 'Prediction error';
            }

            setTimeout(sendFrame, 1000);
        }

        sendFrame();
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    logger.info("Health check called")
    return {"status": "OK"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...), delay: float = 0):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        try:
            with INFERENCE_TIME_HISTOGRAM.time():
                contents = await file.read()
                np_arr = np.frombuffer(contents, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
                gesture = predict(model, le, detector, frame=frame, delay=delay)
                if gesture:
                    MODEL_PREDICTION_TYPES.labels(gesture).inc()
                return {"gesture": gesture or "No hand detected"}
    
        except Exception as e:
            REQUEST_ERRORS.inc()
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail="Could not make prediction")
            

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float  # We ignore this in the backend

class LandmarkRequest(BaseModel):
    landmarks: List[LandmarkPoint]

@app.post("/predict-landmark")
async def predict_landmark(request: LandmarkRequest):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        try:
            if not request.landmarks:
                INVALID_LANDMARK_ERRORS.labels("empty").inc()
                raise HTTPException(status_code=400, detail="No landmarks provided")
            
            if len(request.landmarks) != 21:
                INVALID_LANDMARK_ERRORS.labels("count_mismatch").inc()
                raise HTTPException(status_code=400, detail="Expected 21 landmarks")
                
            with INFERENCE_TIME_HISTOGRAM.time():
                gesture = predict_from_landmarks(model, le, request.landmarks)
                if gesture:
                    MODEL_PREDICTION_TYPES.labels(gesture).inc()
                return {"gesture": gesture or "No gesture detected"}
    
        except Exception as e:
            REQUEST_ERRORS.inc()
            logger.error(f"Landmark prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail="Could not make prediction from landmarks")
        
        

@app.get("/stream")
async def stream_predictions(request: Request, delay: float = 1.0):
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    async def gen_predictions():
        while True:
            if await request.is_disconnected():
                break

            ret, frame = cap.read()
            if not ret:
                yield "data: ERROR\n\n"
                continue

            label = predict(model, le, detector, frame, width, height, delay)
            yield f"data: {label or 'No hand detected'}\n\n"

            await asyncio.sleep(delay)

    return StreamingResponse(gen_predictions(), media_type="text/event-stream")
