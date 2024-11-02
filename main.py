from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = load_model('simple_cnn_emotion_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral", "Contempt"]

def preprocess_image(image_data: bytes, img_size=(48, 48)):
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, img_size)
    normalized_image = resized_image / 255.0
    return normalized_image.reshape(1, img_size[0], img_size[1], 1)

def detect_and_mark_face(image_data: bytes):
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_emotion/")
async def predict_emotion(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()

    marked_image_base64 = detect_and_mark_face(image_data)

    if marked_image_base64 is None:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "No face detected. Please upload an image with a clear face."
        })

    processed_image = preprocess_image(image_data)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_emotion = emotions[predicted_class]

    emotion_descriptions = {
        "Anger": "A strong feeling of annoyance or displeasure.",
        "Disgust": "A feeling of revulsion or profound disapproval.",
        "Fear": "An unpleasant emotion caused by the threat of danger or pain.",
        "Happiness": "A state of well-being and contentment.",
        "Sadness": "A feeling of sorrow or unhappiness.",
        "Surprise": "A feeling of astonishment or shock.",
        "Neutral": "A neutral or unexpressive state.",
        "Contempt": "A feeling that a person or thing is beneath consideration."
    }
    
    emotion_description = emotion_descriptions.get(predicted_emotion, "No description available.")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "emotion": predicted_emotion,
        "emotion_description": emotion_description,
        "image_data": marked_image_base64
    })
