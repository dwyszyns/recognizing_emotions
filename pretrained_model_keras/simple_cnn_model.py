import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

emotions = ["0", "1", "2", "3", "4", "5", "6", "7"]

def create_simple_cnn_model(input_shape, num_classes, learning_rate):
    model = Sequential()

    model.add(Conv2D(6, kernel_size=(6, 6), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(441, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(X_train, y_train, model, batch_size=64, epochs=30):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return model

def test_model(X_test, y_test, model):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    return y_pred, accuracy

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

            # image_resized = augment_image(image_resized)

            landmark_vector = extract_landmarks(image, predictor, detector)
            landmarks.append(landmark_vector)

            X.append(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY))
            y.append(label)
        print(f"Total {emotion} images: {len(image_paths)}")
    
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


if __name__ == "__main__":
    emotions = ["0", "1", "2", "3", "4", "5", "6", "7"]
    X_train, y_train, landmarks_train = load_data_with_landmarks(emotions, "train")
    X_test, y_test, landmarks_test = load_data_with_landmarks(emotions, "test")
    X_train = change_image(landmarks_train, X_train)
    X_test = change_image(landmarks_test, X_test)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = to_categorical(y_train, num_classes=len(emotions))
    y_test = to_categorical(y_test, num_classes=len(emotions))

    input_shape = X_train.shape[1:]
    model = create_simple_cnn_model(input_shape, num_classes=len(emotions), learning_rate=0.001)

    trained_model = train_model(X_train, y_train, model, epochs=20, batch_size=1)

    y_pred, accuracy = test_model(X_test, y_test, trained_model)
    print("CNN Accuracy on test data:", accuracy)
    
    y_pred, accuracy = test_model(X_train, y_train, trained_model)
    print("CNN Accuracy on training data:", accuracy)

    trained_model.save('simple_cnn_emotion_model.h5')
