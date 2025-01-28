from fastapi import FastAPI, File, UploadFile, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import glob
import dlib

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = load_model('model_CNN.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

def preprocess_image(image_data: bytes, img_size:tuple =(48, 48)):
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, img_size)
    normalized_image = resized_image / 255.0
    return normalized_image.reshape(1, img_size[0], img_size[1], 1)

def validate_image(file: UploadFile, allowed_formats=("jpeg", "jpg", "png"), max_size_mb=0.5):
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_formats:
        raise ValueError(f"Invalid file format: {file_extension}. Allowed formats are: {', '.join(allowed_formats)}")
    
    file_size_mb = len(file.file.read()) / (1024 * 1024)
    file.file.seek(0)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File is too large: {file_size_mb:.2f} MB. Maximum allowed size is {max_size_mb} MB.")


def extract_landmarks(image:np.array, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) > 0:  
        shape = predictor(gray, rects[0])
        coords = np.zeros((68, 2), dtype="int")
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    else:
        return np.zeros((68 * 2,))


def detect_and_mark_face(image_data: bytes):
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64

def replace_zero_landmarks(landmarks):
    zero_landmarks = np.zeros((68, 2), dtype=int)
    
    processed_landmarks = [
        landmark if not np.all(landmark == 0) else zero_landmarks
        for landmark in landmarks
    ]
    return np.array(processed_landmarks)

def load_data_with_landmarks(image_data: bytes, img_size:tuple =(48, 48)):
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    resized_image = cv2.resize(image, img_size)
    landmark_vector = extract_landmarks(resized_image, predictor, detector)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    return gray_image, landmark_vector

def change_image(landmarks: np.array, image):
    if np.all(landmarks == 0):
        masked_face = image
        x = image
    else:
        jaw_points = landmarks[0:17]
        forehead_points = landmarks[17:27]

        points = np.concatenate((jaw_points, forehead_points[::-1]), axis=0)

        points = np.array(points, dtype=np.int32)

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        cv2.fillConvexPoly(mask, points, 255)

        if len(image.shape) == 3:
            mask_3ch = cv2.merge([mask, mask, mask])
        else:
            mask_3ch = mask

        result = cv2.bitwise_and(image, mask_3ch)

        background = np.full_like(image, 0)
        masked_face = np.where(mask_3ch == 255, result, background)

        x = cv2.cvtColor(masked_face, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else masked_face
    return x.reshape(-1, 48, 48, 1)

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/analyze-image", response_class=HTMLResponse)
async def analyze_image(request: Request):
    return templates.TemplateResponse("analyze_image.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def analyze_image(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.post("/predict_emotion/")
async def predict_emotion(request: Request, file: UploadFile = File(...)):
    try:
        validate_image(file)
        image_data = await file.read()
        image_marked_face = detect_and_mark_face(image_data)
        gray_image, landmark_vector = load_data_with_landmarks(image_data)

        if gray_image is None:
            if "application/json" in request.headers.get("Accept", ""):
                return JSONResponse(
                    status_code=400,
                    content={"error": "No face detected. Please upload an image with a clear face."}
                )
            else:
                return templates.TemplateResponse(
                    "error.html", 
                    {"request": request, "error_message": "Brak wykrytej twarzy. Proszę przesłać obraz z wyraźną twarzą."}
                )
            
        gray_image = change_image(landmark_vector, gray_image)
        predictions = model.predict(gray_image)
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
        }
        
        emotion_description = emotion_descriptions.get(predicted_emotion, "No description available.")
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "emotion": predicted_emotion,
            "emotion_description": emotion_description,
            "image_data": image_marked_face
        })
    
    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"error": str(e)}
        )
    
@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        image_data = base64.b64decode(data['image_data'])

        gray_image, landmark_vector = load_data_with_landmarks(image_data)
        if gray_image is None or gray_image.size == 0:
            print("Processed image is empty")
            await websocket.send_json({"face_detected": True, "emotion": "Error in processing image"})
            continue
        gray_image = change_image(landmark_vector, gray_image)

        predictions = model.predict(gray_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_emotion = emotions[predicted_class]

        await websocket.send_json({
            "face_detected": True,
            "emotion": predicted_emotion
        })


@app.get("/camera", response_class=HTMLResponse)
async def camera(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request})
