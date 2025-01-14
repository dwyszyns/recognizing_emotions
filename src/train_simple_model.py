import numpy as np
import random
import cv2
import dlib
import glob
from sklearn.metrics import accuracy_score
from scipy.signal import correlate2d
import tensorflow as tf
from datetime import datetime

class Convolution:
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width, _ = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape
        
        self.filter_shape = (num_filters, filter_size, filter_size)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)
        
        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)
        
    def forward(self, input_data):
        self.input_data = input_data
        output = np.zeros(self.output_shape)
        
        for i in range(self.num_filters):
            conv_sum = np.zeros((self.output_shape[1], self.output_shape[2]))
            
            for j in range(self.input_data.shape[2]):
                conv_sum += correlate2d(self.input_data[:, :, j], self.filters[i, :, :], mode="valid")
            
            output[i] = conv_sum + self.biases[i]
        
        output = np.maximum(output, 0)
        return output
    
    def backward(self, dL_dout):
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
            for c in range(self.input_data.shape[0]):
                input_slice = self.input_data[c]

                if input_slice.shape[0] >= dL_dout[i].shape[0] and input_slice.shape[1] >= dL_dout[i].shape[1]:
                    dL_dfilters[i] += correlate2d(input_slice, dL_dout[i], mode="valid")

                    dL_dinput[c] += correlate2d(dL_dout[i], self.filters[i, c], mode="full")

        return dL_dinput, dL_dfilters

class MaxPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        
    def forward(self, input_data):
        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = input_data[c, start_i:end_i, start_j:end_j]
                    self.output[c, i, j] = np.max(patch)

        return self.output
    
    def backward(self, dL_dout):
        dL_dinput = np.zeros_like(self.input_data)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)
                    dL_dinput[c, start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask

        return dL_dinput
    
    
class Fully_Connected:
    def __init__(self, input_size, output_size, adam_lr):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = tf.Variable(np.random.randn(output_size, input_size), dtype=tf.float32)
        self.biases = tf.Variable(np.random.rand(output_size, 1), dtype=tf.float32)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=adam_lr)
        
        self.flattened_input = None
        
    def softmax(self, z):
        shifted_z = z - tf.reduce_max(z)
        exp_values = tf.exp(shifted_z)
        sum_exp_values = tf.reduce_sum(exp_values, axis=0)
        probabilities = exp_values / sum_exp_values
        return probabilities
    
    def softmax_derivative(self, s):
        s = tf.squeeze(s)
        return tf.linalg.diag(s) - tf.linalg.matmul(tf.expand_dims(s, axis=1), tf.expand_dims(s, axis=0))

    
    def forward(self, input_data):
        self.input_data = input_data
        self.flattened_input = tf.convert_to_tensor(input_data.flatten().reshape(1, -1), dtype=tf.float32)  # Zapisz flattened_input
        self.z = tf.matmul(self.weights, self.flattened_input, transpose_b=True) + self.biases
        self.output = self.softmax(self.z)
        return self.output
    
    def backward(self, dL_dout):
        dL_dout = tf.cast(tf.reshape(dL_dout, (self.output_size, 1)), tf.float32)

        dL_dy = tf.linalg.matmul(self.softmax_derivative(self.output), dL_dout)
        dL_dy = tf.reshape(dL_dy, (self.output_size, 1))

        input_data_flat = tf.cast(self.input_data.flatten(), tf.float32)
        dL_dw = tf.linalg.matmul(dL_dy, tf.expand_dims(input_data_flat, axis=0))

        dL_db = dL_dy

        dL_dinput = tf.linalg.matmul(tf.transpose(self.weights), dL_dy)
        dL_dinput = tf.reshape(dL_dinput, self.input_data.shape)

        self.weights.assign_sub(self.optimizer.learning_rate * dL_dw)
        self.biases.assign_sub(self.optimizer.learning_rate * dL_db)
        
        return dL_dinput


def cross_entropy_loss(predictions, targets):
    num_samples = targets.shape[0]
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss

def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples
    return gradient


def train_network(X, y, conv, pool, full, X_test, y_test, lr=0.001, epochs=100):
    epoch_train_losses = []
    epoch_test_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        for i in range(len(X)):
            conv_out = conv.forward(X[i])
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)

            loss = cross_entropy_loss(tf.reshape(full_out, [-1]), tf.convert_to_tensor(y[i], dtype=tf.float32))
            total_loss += loss

            one_hot_pred = tf.one_hot(tf.argmax(full_out), depth=len(y[i]), on_value=1.0, off_value=0.0)
            num_pred = tf.argmax(one_hot_pred)
            num_y = tf.argmax(y[i])

            if tf.reduce_all(tf.equal(num_pred, num_y)):
                correct_predictions += 1
            
            gradient = cross_entropy_loss_gradient(y[i], tf.reshape(full_out, [-1])).numpy()
            full_back = full.backward(tf.convert_to_tensor(gradient, dtype=tf.float32))  
            pool_back = pool.backward(full_back)
            conv_back = conv.backward(pool_back)

        average_loss = total_loss / len(X)
        epoch_train_losses.append(average_loss)
        accuracy = correct_predictions / len(X) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {average_loss:.4f} - Training Accuracy: {accuracy:.2f}%")

        test_loss, test_accuracy = evaluate_network(X_test, y_test, conv, pool, full)
        epoch_test_losses.append(test_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

    return epoch_train_losses, epoch_test_losses


def evaluate_network(X_test, y_test, conv, pool, full):
    total_loss = 0.0
    correct_predictions = 0

    for i in range(len(X_test)):
        conv_out = conv.forward(X_test[i])
        pool_out = pool.forward(conv_out)
        full_out = full.forward(pool_out)

        loss = cross_entropy_loss(tf.reshape(full_out, [-1]), tf.convert_to_tensor(y_test[i], dtype=tf.float32))
        total_loss += loss

        one_hot_pred = tf.one_hot(tf.argmax(full_out), depth=len(y_test[i]), on_value=1.0, off_value=0.0)
        num_pred = tf.argmax(one_hot_pred)
        num_y = tf.argmax(y_test[i])

        if tf.reduce_all(tf.equal(num_pred, num_y)):
            correct_predictions += 1

    average_loss = total_loss / len(X_test)
    accuracy = correct_predictions / len(X_test) * 100.0

    return average_loss, accuracy



def predict(input_sample, conv, pool, full):
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    flattened_output = pool_out.flatten()
    predictions = full.forward(flattened_output)
    return predictions


def augment_image(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    angle = random.randint(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    value = random.randint(-30, 30)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    alpha = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    image = cv2.add(image, noise)

    return image

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

def calculate_accuracy(X, y, conv, pool, full):
    predictions = []
    for data in X:
        pred = predict(data, conv, pool, full)
        one_hot_pred = np.zeros_like(pred)
        one_hot_pred[np.argmax(pred)] = 1
        predictions.append(one_hot_pred.flatten())
    
    predictions = np.array(predictions)
    accuracy = accuracy_score(predictions, y) * 100
    return accuracy

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


def godzina():
    aktualna_godzina = datetime.now()
    sformatowana_godzina = aktualna_godzina.strftime("%H:%M:%S")
    print("Aktualna godzina:", sformatowana_godzina)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    emotions = ["0", "1", "2", "3", "4", "5", "6", "7"]
    X_train, y_train, landmarks_train = load_data_with_landmarks(emotions, "RAF-DB/train")
    X_test, y_test, landmarks_test = load_data_with_landmarks(emotions, "RAF-DB/test")
    X_train = change_image(landmarks_train, X_train)
    X_test = change_image(landmarks_train, X_test)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = np.eye(len(emotions))[y_train]
    y_test = np.eye(len(emotions))[y_test]
    
    input_shape = X_train.shape[1:]
    print(input_shape)
    
    all_epoch_train_losses = []
    all_epoch_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    runs=6
    for run in range(runs):
        print(f"Training run {run + 1}/{runs}")

        conv = Convolution(X_train[0].shape, 6, 1)
        pool = MaxPool(2)
        full = Fully_Connected(441, 8, adam_lr=0.01)
        godzina()

        epoch_train_losses, epoch_test_losses = train_network(X_train, y_train, conv, pool, full, X_test, y_test, lr=0.001, epochs=20)
        godzina()
        all_epoch_train_losses.append(epoch_train_losses)
        all_epoch_test_losses.append(epoch_test_losses)

        train_accuracy = calculate_accuracy(X_train, y_train, conv, pool, full)
        print(f"Train accuracy: {train_accuracy}")
        test_accuracy = calculate_accuracy(X_test, y_test, conv, pool, full)
        print(f"Test accuracy: {test_accuracy}")
        all_train_accuracies.append(train_accuracy)
        all_test_accuracies.append(test_accuracy)
    
    avg_epoch_train_losses = np.mean(all_epoch_train_losses, axis=0)
    avg_epoch_test_losses = np.mean(all_epoch_test_losses, axis=0)

    with open("RAFDBtraining_results_0,001_epochs=20_runs=6.txt", "w") as f:
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

        avg_train_accuracy = round(np.mean(all_train_accuracies)*100, 4)
        avg_test_accuracy = round(np.mean(all_test_accuracies)*100, 4)
        f.write("\nAverage Train Accuracy: {:.2f}%\n".format(avg_train_accuracy))
        f.write("Average Test Accuracy: {:.2f}%\n".format(avg_test_accuracy))

    print("Training complete. Results saved to training_results.txt")