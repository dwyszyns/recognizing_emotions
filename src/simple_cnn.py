import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from scipy.signal import correlate2d
import tensorflow as tf
from functions import make_plot_losses_per_epochs, preprocess_images_from_dataset, scale_and_one_hot_encode


class Layer(ABC):
    @abstractmethod
    def forward(self, input_data):
        """
        Perform the forward pass for the layer.
        :param input_data: Input data for the layer.
        :return: Output of the layer after the forward pass.
        """
        pass

    @abstractmethod
    def backward(self, dL_dout):
        """
        Perform the backward pass for the layer.
        :param dL_dout: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        pass


class Convolution_Layer(Layer):
    def __init__(self, input_shape, filter_size, num_filters, activation="relu"):
        self.filters_number = num_filters
        self.weights = np.random.randn(num_filters, filter_size, filter_size)
        self.output_dims = (num_filters, input_shape[0] - filter_size + 1, input_shape[1] - filter_size + 1)
        self.activation = activation
        self.biases = np.random.randn(*self.output_dims)

    def forward(self, input_data):
        self.input_data = input_data
        feature_map = np.zeros(self.output_dims)
        
        for filter_idx in range(self.filters_number):
            filtered_output = np.zeros((self.output_dims[1], self.output_dims[2]))
            
            for channel_idx in range(self.input_data.shape[2]):
                input_slice = self.input_data[:, :, channel_idx]
                kernel = self.weights[filter_idx, :, :]
                filtered_output += self.apply_filter(input_slice, kernel)
                
            feature_map[filter_idx] = filtered_output + self.biases[filter_idx]
            
        feature_map = self.activate(feature_map)
        return feature_map

    def apply_filter(self, input_slice, kernel, mode="valid"):
        return correlate2d(input_slice, kernel, mode=mode)

    def activate(self, feature_map):
        if self.activation == "relu":
            return np.maximum(feature_map, 0)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-feature_map))
        return feature_map

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input_data)
        grad_kernels = np.zeros_like(self.weights)

        for filter_idx in range(self.filters_number):
            for channel_idx in range(self.input_data.shape[0]):
                input_slice = self.input_data[channel_idx]
                if self.is_valid_slice(input_slice, grad_output[filter_idx]):
                    grad_kernels[filter_idx] += self.apply_filter(input_slice, grad_output[filter_idx])
                    grad_input[channel_idx] += self.apply_filter(grad_output[filter_idx], self.weights[filter_idx, channel_idx], mode="full")

        return grad_input, grad_kernels

    def is_valid_slice(self, region, gradient):
        return region.shape[0] >= gradient.shape[0] and region.shape[1] >= gradient.shape[1]


class MaxPool_Layer(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def iterate_regions(self, input_data):
        for channel_idx in range(self.num_channels):
            for row in range(self.output_height):
                for col in range(self.output_width):
                    start_row = row * self.pool_size
                    start_col = col * self.pool_size
                    end_row = start_row + self.pool_size
                    end_col = start_col + self.pool_size

                    region = input_data[channel_idx, start_row:end_row, start_col:end_col]
                    yield channel_idx, row, col, region

    def initialize_dimensions(self, input_tensor):
        self.input_tensor = input_tensor
        self.num_channels, self.input_height, self.input_width = input_tensor.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size
        
    def forward(self, input_tensor):
        self.initialize_dimensions(input_tensor)
        pooled_output = np.zeros((self.num_channels, self.output_height, self.output_width))
        for channel_idx, i, j, region in self.iterate_regions(input_tensor):
            pooled_output[channel_idx, i, j] = np.max(region)
        return pooled_output

    def backward(self, gradient_output):
        gradient_input = np.zeros_like(self.input_tensor)
        for channel_idx, i, j, region in self.iterate_regions(self.input_tensor):
            pool_mask = region == np.max(region)
            gradient_input[channel_idx, i * self.pool_size:(i + 1) * self.pool_size,
                           j * self.pool_size:(j + 1) * self.pool_size] = gradient_output[channel_idx, i, j] * pool_mask
        return gradient_input
    
    
class Fully_Connected_Layer(Layer):
    def __init__(self, input_size, output_size, adam_lr):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=adam_lr)
        self.weights = tf.Variable(np.random.randn(output_size, input_size), dtype=tf.float32)
        self.biases = tf.Variable(np.random.rand(output_size, 1), dtype=tf.float32)
        self.flattened_input = None

    def softmax(self, logits):
        shifted_logits = logits - tf.reduce_max(logits)
        exp_values = tf.exp(shifted_logits)
        total_exp_values = tf.reduce_sum(exp_values, axis=0)
        softmax_output = exp_values / total_exp_values
        return softmax_output

    def softmax_derivative(self, softmax_output):
        s_output = tf.squeeze(softmax_output)
        return tf.linalg.diag(s_output) - tf.linalg.matmul(tf.expand_dims(s_output, axis=1), tf.expand_dims(s_output, axis=0))

    def forward(self, input_data):
        self.input_data = input_data
        self.flattened_input_data = tf.convert_to_tensor(input_data.flatten().reshape(1, -1), dtype=tf.float32)
        self.linear_combination = tf.matmul(self.weights, self.flattened_input_data, transpose_b=True) + self.biases
        self.model_output = self.softmax(self.linear_combination)
        return self.model_output

    def backward(self, grad_output):
        grad_activation = self.compute_grad_activation(grad_output)
        grad_weights, grad_biases = self.compute_grad_weights_and_biases(grad_activation)
        grad_input_data = self.compute_grad_input_data(grad_activation)
        self.update_parameters(grad_weights, grad_biases)
        return grad_input_data

    def compute_grad_activation(self, grad_output):
        grad_output = tf.cast(tf.reshape(grad_output, (self.output_size, 1)), tf.float32)
        grad_activation = tf.linalg.matmul(self.softmax_derivative(self.model_output), grad_output)
        return tf.reshape(grad_activation, (self.output_size, 1))

    def compute_grad_weights_and_biases(self, grad_activation):
        flattened_input_data = tf.cast(self.input_data.flatten(), tf.float32)
        grad_weights = tf.linalg.matmul(grad_activation, tf.expand_dims(flattened_input_data, axis=0))
        grad_biases = grad_activation
        return grad_weights, grad_biases

    def compute_grad_input_data(self, grad_activation):
        grad_input_data = tf.linalg.matmul(tf.transpose(self.weights), grad_activation)
        return tf.reshape(grad_input_data, self.input_data.shape)

    def update_parameters(self, grad_weights, grad_biases):
        self.weights.assign_sub(self.optimizer.learning_rate * grad_weights)
        self.biases.assign_sub(self.optimizer.learning_rate * grad_biases)


def cross_entropy_loss(predicted_probs, true_labels, epsilon=1e-7):
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    num_samples = true_labels.shape[0]
    loss = -np.sum(true_labels * np.log(predicted_probs)) / num_samples
    return loss


def cross_entropy_loss_gradient(true_labels, predicted_probs, epsilon=1e-7):
    num_samples = true_labels.shape[0]
    gradient = -true_labels / (predicted_probs + epsilon) / num_samples
    return gradient


def create_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def train_network(X, y, conv, pool, full, X_test, y_test, epochs=100, batch_size=20):
    epoch_train_losses = []
    epoch_test_losses = []
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        correct_predictions = 0
        batches = list(create_batches(X, y, batch_size))
        X_batches, y_batches = zip(*batches)
        for batch_idx, batch in enumerate(batches):
            for sample_idx in range(len(batch)):
                conv_out = conv.forward(X_batches[batch_idx][sample_idx])
                pool_out = pool.forward(conv_out)
                full_out = full.forward(pool_out)

                batch_loss = cross_entropy_loss(tf.reshape(full_out, [-1]), tf.convert_to_tensor(y_batches[batch_idx][sample_idx], dtype=tf.float32))
                total_epoch_loss += batch_loss

                predicted_label = tf.one_hot(tf.argmax(full_out), depth=len(y_batches[batch_idx][sample_idx]), on_value=1.0, off_value=0.0)
                predicted_class = tf.argmax(predicted_label)
                true_class = tf.argmax(y_batches[batch_idx][sample_idx])

                if tf.reduce_all(tf.equal(predicted_class, true_class)):
                    correct_predictions += 1
                
                gradient = cross_entropy_loss_gradient(y_batches[batch_idx][sample_idx], tf.reshape(full_out, [-1])).numpy()
                full_back = full.backward(tf.convert_to_tensor(gradient, dtype=tf.float32))  
                pool_back = pool.backward(full_back)
                conv_back = conv.backward(pool_back)

        average_loss = total_epoch_loss / len(X)
        epoch_train_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {average_loss:.4f}")

        test_loss, _ = evaluate_network(X_test, y_test, conv, pool, full)
        epoch_test_losses.append(test_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f}")

    return epoch_train_losses, epoch_test_losses


def train_network_without_batches(X, y, conv, pool, full, X_test, y_test, epochs=100):
    epoch_train_losses = []
    epoch_test_losses = []
    for epoch in range(epochs):
        total_epoch_loss = 0.0
        correct_predictions = 0
        for sample_idx in range(len(X)):
            conv_out = conv.forward(X[sample_idx])
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)

            loss = cross_entropy_loss(tf.reshape(full_out, [-1]), tf.convert_to_tensor(y[sample_idx], dtype=tf.float32))
            total_epoch_loss += loss

            predicted_label = tf.one_hot(tf.argmax(full_out), depth=len(y[sample_idx]), on_value=1.0, off_value=0.0)
            predicted_class = tf.argmax(predicted_label)
            true_class = tf.argmax(y[sample_idx])

            if tf.reduce_all(tf.equal(predicted_class, true_class)):
                correct_predictions += 1
            
            gradient = cross_entropy_loss_gradient(y[sample_idx], tf.reshape(full_out, [-1])).numpy()
            full_back = full.backward(tf.convert_to_tensor(gradient, dtype=tf.float32))  
            pool_back = pool.backward(full_back)
            conv_back = conv.backward(pool_back)

        average_loss = total_epoch_loss / len(X)
        epoch_train_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {average_loss:.4f} ")

        test_loss, _ = evaluate_network(X_test, y_test, conv, pool, full)
        epoch_test_losses.append(test_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f}")

    return epoch_train_losses, epoch_test_losses


def evaluate_network(X_test, y_test, conv, pool, full):
    total_epoch_loss = 0.0
    correct_predictions = 0

    for sample_idx in range(len(X_test)):
        conv_out = conv.forward(X_test[sample_idx])
        pool_out = pool.forward(conv_out)
        full_out = full.forward(pool_out)

        loss = cross_entropy_loss(tf.reshape(full_out, [-1]), tf.convert_to_tensor(y_test[sample_idx], dtype=tf.float32))
        total_epoch_loss += loss

        predicted_label = tf.one_hot(tf.argmax(full_out), depth=len(y_test[sample_idx]), on_value=1.0, off_value=0.0)
        predicted_class = tf.argmax(predicted_label)
        true_class = tf.argmax(y_test[sample_idx])

        if tf.reduce_all(tf.equal(predicted_class, true_class)):
            correct_predictions += 1

    average_loss = total_epoch_loss / len(X_test)
    accuracy = correct_predictions / len(X_test) * 100.0

    return average_loss, accuracy


def predict(input_data, convolutional_layer, pooling_layer, dense_layer):
    conv_result = convolutional_layer.forward(input_data)
    pooled_result = pooling_layer.forward(conv_result)
    flattened_result = pooled_result.flatten()
    final_predictions = dense_layer.forward(flattened_result)
    return final_predictions


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


def train_single_run(labels, X_train, y_train, X_test, y_test, epochs=20, batch_size=16, learning_rate=0.1):
    conv = Convolution_Layer(X_train[0].shape, 6, 1)
    pool = MaxPool_Layer(2)
    full = Fully_Connected_Layer(441, len(labels), adam_lr=learning_rate)

    epoch_train_losses, epoch_test_losses = train_network(X_train, y_train, conv, pool, full, X_test, y_test, 
                                                              epochs, batch_size)
    train_accuracy = calculate_accuracy(X_train, y_train, conv, pool, full)
    print(f"Train accuracy: {train_accuracy}")
    test_accuracy = calculate_accuracy(X_test, y_test, conv, pool, full)
    print(f"Test accuracy: {test_accuracy}")
    
    return train_accuracy, test_accuracy, epoch_train_losses, epoch_test_losses


def run_cnn_experiments(dataset, runs, labels):
    X_train, y_train, X_test, y_test = dataset
    all_epoch_train_losses = []
    all_epoch_val_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    print("----------Training phase----------")
    for run in range(runs):
        print(f"Training run {run + 1}/{runs}")
        train_accuracy, test_accuracy, train_losses, val_losses = train_single_run(labels, X_train, y_train, X_test, y_test)
        all_epoch_train_losses.append(train_losses)
        all_epoch_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracy)
        all_test_accuracies.append(test_accuracy)

    avg_train_accuracy = round(np.mean(all_train_accuracies), 2)
    avg_test_accuracy = round(np.mean(all_test_accuracies), 2)
    print(f"Average train accuracy: {avg_train_accuracy}%")
    print(f"Average test accuracy: {avg_test_accuracy}%")
    
    return all_epoch_train_losses, all_epoch_val_losses


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    dataset = preprocess_images_from_dataset(labels)
    dataset = scale_and_one_hot_encode(dataset, len(labels))
    
    train_losses, test_losses = run_cnn_experiments(dataset, 1, labels)
    make_plot_losses_per_epochs(train_losses)
    