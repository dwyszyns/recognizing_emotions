# Emotion Recognition Application

A web application written in Python using the FastAPI framework for recognizing emotions based on images and live camera feed. The user can also train a custom machine learning model on provided data. This tool is dedicated to analyzing emotions and experimenting with emotion recognition algorithms.

---

# Table of Contents

1. [Requirements and Installation](#requirements-and-installation)  
    - [Prerequisites](#prerequisites)  
    - [Installation Steps](#installation-steps)  
2. [Running the Application](#running-the-application)  
    - [Running the Service](#running-the-service)  
    - [Training a Custom Model](#training-a-custom-model)  
3. [Project Structure](#project-structure)

---

## Requirements and Installation

### Prerequisites

- **Python**: 3.10.12
- **Conda** or **venv** (for virtual environments)
    ```bash
    sudo apt-get install build-essential cmake
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libboost-all-dev
    ```

---

### Installation Steps

#### 1. Create a virtual environment:

- **Using Conda:**
  ```bash
  conda create --name myenv python=3.10.12
  conda activate myenv
  ```

- **Using venv:**
  ```bash
  python3.10 -m venv myenv
  source myenv/bin/activate  # Linux/Mac
  myenv\Scripts\activate     # Windows
  ```

#### 2. Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

#### 3. Clone the repository:
  ```bash
  git clone https://github.com/dwyszyns/recognizing_emotions.git
  ```

---

## Running the Application

### Running the Service

To run the service for recognizing emotions from images and the camera feed, execute the following command in the terminal:
  ```bash
  uvicorn main:app --reload
  ```

### Training a Custom Model

1. Download the appropriate dataset and divide it into two directories: `train` and `test`.
2. Run one of the methods available in the `src` folder to train the model.

#### Highest Achieved Average Accuracy Results - Using the algorithm from `complex_cnn.py`:
- **RAF-DB:**
  - Test Data: 56%
  - Training Data: 66%
- **CK+:**
  - Test Data: 81%
  - Training Data: 90%

---

## Project Structure

- **src**  
  Contains machine learning model code used for training:
  - `adaboost_algorithm.py` – Implementation of the AdaBoost model.
  - `random_forest_algorithm.py` – Implementation of the Random Forest model.
  - `complex_cnn.py` – Implementation of complex convolutional networks using TensorFlow.
  - `simple_cnn.py` – Implementation of a simplified version of convolutional networks written from scratch.
  - `functions.py` – Functions used during the training process. These include image preprocessing and feature extraction functions.

- **static**  
  Contains static resources used in the application:
  - `css` folder: Includes the `style.css` file that defines the application style.
  - Images: Pictures used in the application, e.g., examples or logos.

- **templates**  
  Folder containing HTML templates for the FastAPI web application. These include: 
  - `index.html` – Main page template.
  - `analyze_image.html` – Template for uploading an image for emotion analysis.
  - `result.html` – Template displaying the predicted emotion and the analyzed image.
  - `camera.html` – Template for analyzing emotions from the camera feed.
  - `about.html` – Template with general information about the application and its author.
  - `error.html` – Template for error messages when an issue occurs with the uploaded image or analysis.

- **tests**  
  Folder containing tests:
  - `img` folder: Contains images used for testing the application.
  - `test_main.py`: Script containing unit, integration, validation, system, and acceptance tests for `main.py`.

- **README.md**  
  Project documentation, including information on installation, usage, and application objectives.

- **haarcascade_frontalface_default.xml**  
  File used for face detection with the Haar Cascade algorithm.

- **main.py**  
  The main code for the FastAPI application:
  - Defines all endpoints.
  - Handles image processing and application logic.

- **model_CNN.h5**  
  Pretrained convolutional neural network (CNN) model based on the RAF-DB dataset, created using the code in `complex_cnn.py`.

- **requirements.txt**  
  File listing the Python libraries required for the project to run.

