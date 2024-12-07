import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback

emotions = ["0", "1", "2", "3", "4", "5", "6", "7"]

def create_simple_cnn_model(input_shape: tuple, num_classes: int, learning_rate: float):
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

def train_model(X_train: np.array, y_train: np.array, model:Sequential, batch_size: int =64, epochs: int =30):
    print_lr_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: print(f"Learning Rate at epoch {epoch+1}: {model.optimizer.lr.numpy()}"))
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[print_lr_callback])
    
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    
    return model, train_losses, val_losses


def test_model(X_test: np.array, y_test: np.array, model: Sequential):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    return y_pred, accuracy

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


def load_data_with_landmarks(emotion_list: list, base_dir: str ="train", img_size: tuple =(48, 48)):
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
    
    X = np.array(X)
    y = np.array(y)
    landmarks = np.array(landmarks)
    
    return X, y, landmarks

def change_image(landmarks: np.array, images: np.array):
    colored_images = []
    for i in range(len(images)):
        jaw_points = landmarks[i][0:17]
        forehead_points = landmarks[i][17:27]

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
    X_test = X_test.reshape(-1, 48, 48, 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = to_categorical(y_train, num_classes=len(emotions))
    y_test = to_categorical(y_test, num_classes=len(emotions))

    input_shape = X_train.shape[1:]
    all_epoch_train_losses = []
    all_epoch_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    runs=6
    for run in range(runs):
        print(f"Training run {run + 1}/{runs}")
        model = create_simple_cnn_model(input_shape, num_classes=len(emotions), learning_rate=0.001)

        trained_model, train_losses, val_losses = train_model(X_train, y_train, model, epochs=20, batch_size=1)

        all_epoch_train_losses.append(train_losses)
        all_epoch_test_losses.append(val_losses)
        
        y_pred, test_accuracy = test_model(X_test, y_test, trained_model)
        print("CNN Accuracy on test data:", test_accuracy)
        y_pred, train_accuracy = test_model(X_train, y_train, trained_model)
        print("CNN Accuracy on training data:", train_accuracy)
        all_train_accuracies.append(train_accuracy)
        all_test_accuracies.append(test_accuracy)

        trained_model.save(f"cnn_model_{run}.h5")
    
    avg_epoch_train_losses = np.mean(all_epoch_train_losses, axis=0)
    avg_epoch_test_losses = np.mean(all_epoch_test_losses, axis=0)
    
    with open("cnn_results.txt", "w") as f:
        f.write("Average training loss per epoch across all runs:\n")
        for epoch, loss in enumerate(avg_epoch_train_losses, 1):
            f.write(f"Epoch {epoch}: {loss:.4f}\n")
        
        f.write("Average validating loss per epoch across all runs:\n")
        for epoch, loss in enumerate(avg_epoch_test_losses, 1):
            f.write(f"Epoch {epoch}: {loss:.4f}\n")
        
        f.write("\nTrain and Test Accuracy for each run:\n")
        for run in range(runs):
            f.write(f"Run {run + 1} - Train Accuracy: {all_train_accuracies[run]:.2f}%, "
                    f"Test Accuracy: {all_test_accuracies[run]:.2f}%\n")

        avg_train_accuracy = np.mean(all_train_accuracies)
        avg_test_accuracy = np.mean(all_test_accuracies)
        f.write("\nAverage Train Accuracy: {:.2f}%\n".format(avg_train_accuracy))
        f.write("Average Test Accuracy: {:.2f}%\n".format(avg_test_accuracy))

    print("Training complete. Results saved to training_results.txt")
