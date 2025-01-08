import numpy as np
import cv2
import glob
import dlib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def extract_landmarks(image, predictor, detector):
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

def replace_zero_landmarks(landmarks):
    zero_landmarks = np.zeros((68, 2), dtype=int)
    
    processed_landmarks = [
        landmark if not np.all(landmark == 0) else zero_landmarks
        for landmark in landmarks
    ]
    return np.array(processed_landmarks)

def load_data_with_landmarks(emotion_list, base_dir="train", img_size=(48, 48)):
    X = []
    y = []
    landmarks = []
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    for emotion in emotion_list:
        image_paths = glob.glob(f"{base_dir}/{emotion}/*")
        label = emotion_list.index(emotion) 
        for path in image_paths:
            image = cv2.imread(path)
            image_resized = cv2.resize(image, img_size) 
            landmark_vector = extract_landmarks(image, predictor, detector)
            landmarks.append(landmark_vector)

            X.append(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY))
            y.append(label)
        print(f"Total {emotion} images: {len(image_paths)}")
    landmarks = replace_zero_landmarks(landmarks)
    X = np.array(X)
    y = np.array(y)
    landmarks = np.array(landmarks)
    
    return X, y, landmarks

def change_image(landmarks, images):
    colored_images = []
    for i in range(len(images)):
        jaw_points = landmarks[i][0:17]  # Punkty żuchwy
        forehead_points = landmarks[i][17:27]  # Punkty górnej części twarzy

        points = np.concatenate((jaw_points, forehead_points[::-1]), axis=0)

        points = np.array(points, dtype=np.int32)

        image = images[i] 
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        cv2.fillConvexPoly(mask, points, 255)

        if len(image.shape) == 3:
            mask_3ch = cv2.merge([mask, mask, mask])
        else:
            mask_3ch = mask

        result = cv2.bitwise_and(image, mask_3ch)

        background = np.full_like(image, 0)
        masked_face = np.where(mask_3ch == 255, result, background)

        colored_images.append(cv2.cvtColor(masked_face, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else masked_face)
    return np.array(colored_images).reshape(-1, 48, 48, 1)

def accuracy_class_wise(y_true, y_pred):
    class_accuracy = {}
    unique_classes = np.unique(y_true)
    
    for class_label in unique_classes:
        indices = np.where(y_true == class_label)[0]
        class_predictions = y_pred[indices]
        class_true_labels = y_true[indices]
        accuracy = np.mean(class_predictions == class_true_labels)
        class_accuracy[class_label] = accuracy
    
    return class_accuracy

def train(random_forest, it_num, X_train, y_train, X_test, y_test):
    scores_test = []
    scores_train = []
    models = []

    for _ in range(it_num):
        random_forest.fit(X_train, y_train)
        y_pred_test = random_forest.predict(X_test)
        y_pred_train = random_forest.predict(X_train)
        
        scores_test.append(accuracy_score(y_test, y_pred_test))
        scores_train.append(accuracy_score(y_train, y_pred_train))
        models.append(random_forest)

    avg_train_accuracy = np.mean(scores_train)
    avg_test_accuracy = np.mean(scores_test)
    
    return scores_train, scores_test, models, avg_train_accuracy, avg_test_accuracy

def show_confusion_matrix(best_model, X_test, y_test):
    best_y_pred_test = best_model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, best_y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

    plt.title('Macierz pomyłek - najlepszy model', fontsize=16)
    plt.xlabel('Przewidziane etykiety', fontsize=12)
    plt.ylabel('Prawdziwe etykiety', fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    emotions = ["0", "1", "2", "3", "4", "5", "6", "7"]

    X_train, y_train, landmarks_train = load_data_with_landmarks(emotions, "RAF-DB/train")
    X_test, y_test, landmarks_test = load_data_with_landmarks(emotions, "RAF-DB/test")
    X_train = change_image(landmarks_train, X_train)
    X_test = change_image(landmarks_train, X_test)

    pca = PCA(n_components=100)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    rf = RandomForestClassifier(criterion='gini', max_depth=7, max_features='sqrt', min_samples_leaf=1, n_estimators=50)
    
    scores_train, scores_test, models, avg_train_accuracy, avg_test_accuracy = train(rf, 5, X_train_pca, y_train, X_test_pca, y_test)

    best_model_index = np.argmax(scores_test)
    best_model = models[best_model_index]
    
    show_confusion_matrix(best_model, X_test_pca, y_test)

    