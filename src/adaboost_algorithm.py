import numpy as np
import cv2
import glob
import dlib

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier


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

def train(model, it_num, X_train, y_train, X_test, y_test):
    scores_test = []
    scores_train = []
    models = []

    for _ in range(it_num):
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        scores_test.append(accuracy_score(y_test, y_pred_test))
        scores_train.append(accuracy_score(y_train, y_pred_train))
        models.append(model)

    avg_train_accuracy = np.mean(scores_train)
    avg_test_accuracy = np.mean(scores_test)
    
    return models, avg_train_accuracy, avg_test_accuracy

if __name__ == "__main__":
    emotions = ["0", "1", "2", "3", "4", "5", "6", "7"]

    X_train, y_train, landmarks_train = load_data_with_landmarks(emotions, "train")
    X_test, y_test, landmarks_test = load_data_with_landmarks(emotions, "test")
    X_train = change_image(landmarks_train, X_train)
    X_test = change_image(landmarks_train, X_test)

    pca = PCA(n_components=100)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    ada_model = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=30, learning_rate=1.0)
    
    models, avg_train_accuracy, avg_test_accuracy = train(ada_model, 5, X_train_pca, y_train, X_test_pca, y_test)

    print(avg_test_accuracy)

    