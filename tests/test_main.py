import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# --- Testy Walidacyjne ---
@pytest.mark.parametrize(
    "file_path, expected_status_code, expected_error_message",
    [
        ("tests/image1.jpg", 200, None),
        ("tests/not_image.txt", 422, "Invalid file format"),
        ("tests/large_image.jpg", 422, "File is too large"),
    ]
)
def test_file_validation(file_path, expected_status_code, expected_error_message):
    with open(file_path, "rb") as file:
        response = client.post("/predict_emotion/", files={"file": file})
    assert response.status_code == expected_status_code
    if expected_error_message:
        assert expected_error_message in response.text

# --- Testy Akceptacyjne ---
@pytest.mark.parametrize(
    "file_path, expected_emotions",
    [
        ("tests/neutral_face.jpg", ["Neutral"]),
        ("tests/happy_face.jpg", ["Happiness"]),
        ("tests/sad_face.jpg", ["Sadness", "Fear"]),
        ("tests/surprise_face.jpg", ["Anger", "Surprise"]),
        ("tests/disgust_face.jpg", ["Disgust", "Fear"]),
    ]
)
def test_emotion_prediction(file_path, expected_emotions):
    with open(file_path, "rb") as img:
        response = client.post("/predict_emotion/", files={"file": img})
    assert response.status_code == 200
    assert "emotion" in response.text
    assert any(emotion in response.text for emotion in expected_emotions)

# --- Testy Funkcjonalne ---
@pytest.mark.parametrize(
    "file_path, expected_status_code, expected_response_text",
    [
        ("tests/black_photo.PNG", 400, "No face detected"),
        ("tests/group_photo.jpg", 200, "emotion"),
    ]
)
def test_image_functionality(file_path, expected_status_code, expected_response_text):
    with open(file_path, "rb") as img:
        response = client.post("/predict_emotion/", files={"file": img})
    assert response.status_code == expected_status_code
    assert expected_response_text in response.text

# --- Testy Stron HTML ---
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
