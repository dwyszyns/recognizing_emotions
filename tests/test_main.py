import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from main import app, validate_image, preprocess_image, detect_and_mark_face
import io
import cv2
import numpy as np

client = TestClient(app)

def test_integration_valid_image():
    with open("tests/img/image1.jpg", "rb") as img:
        response = client.post("/predict_emotion/", files={"file": img})
    assert response.status_code == 200
    assert "emotion" in response.text

def test_integration_invalid_format():
    with open("tests/img/not_image.txt", "rb") as txt:
        response = client.post("/predict_emotion/", files={"file": txt})
    assert response.status_code == 422
    assert "Invalid file format" in response.text

def test_integration_large_file():
    with open("tests/img/large_image.jpg", "rb") as large_file:
        response = client.post("/predict_emotion/", files={"file": large_file})
    assert response.status_code == 422
    assert "File is too large" in response.text


@pytest.mark.parametrize(
    "file_path, expected_emotions",
    [
        ("tests/img/neutral_face.jpg", ["Neutral"]),
        ("tests/img/happy_face.jpg", ["Happiness"]),
        ("tests/img/sad_face.jpg", ["Sadness", "Fear"]),
        ("tests/img/surprise_face.jpg", ["Anger", "Surprise"]),
        ("tests/img/disgust_face.jpg", ["Disgust", "Fear", "Neutral"]),
    ]
)
def test_emotion_prediction(file_path, expected_emotions):
    with open(file_path, "rb") as img:
        response = client.post("/predict_emotion/", files={"file": img})
    assert response.status_code == 200
    assert "emotion" in response.text
    assert any(emotion in response.text for emotion in expected_emotions)


@pytest.mark.parametrize(
    "file_path, expected_status_code, expected_response_text",
    [
        ("tests/img/black_photo.PNG", 200, "Try Another Image"),
        ("tests/img/group_photo.jpg", 200, "emotion"),
    ]
)
def test_image_functionality(file_path, expected_status_code, expected_response_text):
    with open(file_path, "rb") as img:
        response = client.post("/predict_emotion/", files={"file": img}, headers = {"Accept": "application/json"})
    assert response.status_code == expected_status_code
    assert expected_response_text in response.text
    
def test_validate_image_valid_file():
    valid_file = UploadFile(filename="tests/img/image1.jpg", file=io.BytesIO(b"valid image content"))
    valid_file.file.seek(0)
    try:
        validate_image(valid_file)
    except Exception:
        pytest.fail("validate_image raised an exception unexpectedly!")
        
def test_preprocess_image():
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White image
    _, buffer = cv2.imencode('.jpg', test_image)
    preprocessed_image = preprocess_image(buffer.tobytes())
    assert preprocessed_image.shape == (1, 48, 48, 1)
    assert np.max(preprocessed_image) <= 1.0  # Normalized

def test_detect_and_mark_face_no_faces():
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image
    _, buffer = cv2.imencode('.jpg', test_image)
    marked_image = detect_and_mark_face(buffer.tobytes())
    assert marked_image is None
    

@pytest.mark.parametrize(
    "endpoint, expected_status_code, expected_content",
    [
        ("/", 200, "Home"),
        ("/analyze-image", 200, "Upload"),
        ("/camera", 200, "Emotion Detection from Camera"),
        ("/about", 200, "About Us"),
    ]
)
def test_html_pages(endpoint, expected_status_code, expected_content):
    response = client.get(endpoint)
    assert response.status_code == expected_status_code
    assert expected_content in response.text
