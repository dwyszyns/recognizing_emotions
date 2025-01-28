import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from functions import make_plot_losses_per_epochs, preprocess_images_from_dataset, scale_and_one_hot_encode


def create_cnn_model(input_shape: tuple, num_classes: int, learning_rate: float, conv_layers: int=6, filters: list=[32, 64, 128, 256]):
    model = Sequential()

    model.add(Conv2D(filters[0], kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(1, conv_layers):
        model.add(Conv2D(filters[min(i, len(filters) - 1)], kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  
    model.add(Dense(num_classes, activation='softmax')) 

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def train_model(X_train: np.array, y_train: np.array, model:Sequential, batch_size: int =64, epochs: int =30):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    
    return model, train_losses, val_losses


def test_model(X_test: np.array, y_test: np.array, model: Sequential):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    return y_pred, accuracy


def train_single_run(input_shape: tuple, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array):
    model = create_cnn_model(input_shape,
                                num_classes=len(labels),
                                learning_rate=0.001,
                                conv_layers=2,
                                filters=[32, 64, 128, 256])

    trained_model, train_losses, val_losses = train_model(X_train,
                                                          y_train,
                                                          model,
                                                          epochs=10, 
                                                          batch_size=64)
    
    _, test_accuracy = test_model(X_test, y_test, trained_model)
    print("CNN Accuracy on test data:", test_accuracy)
    _, train_accuracy = test_model(X_train, y_train, trained_model)
    print("CNN Accuracy on training data:", train_accuracy)
    
    return train_accuracy, test_accuracy, train_losses, val_losses
    

def run_cnn_experiments(dataset: tuple, runs: int):
    X_train, y_train, X_test, y_test = dataset
    input_shape = X_train.shape[1:]
    all_epoch_train_losses = []
    all_epoch_val_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    print("----------Training phase----------")
    for run in range(runs):
        print(f"Training run {run + 1}/{runs}")
        train_accuracy, test_accuracy, train_losses, val_losses = train_single_run(input_shape, X_train, y_train, X_test, y_test)
        all_epoch_train_losses.append(train_losses)
        all_epoch_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracy)
        all_test_accuracies.append(test_accuracy)

    avg_train_accuracy = round(np.mean(all_train_accuracies)*100, 2)
    avg_test_accuracy = round(np.mean(all_test_accuracies)*100, 2)
    print(f"Average train accuracy: {avg_train_accuracy}%")
    print(f"Average test accuracy: {avg_test_accuracy}%")
    
    return all_epoch_train_losses, all_epoch_val_losses


if __name__ == "__main__":
    labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    dataset = preprocess_images_from_dataset(labels)
    dataset = scale_and_one_hot_encode(dataset, len(labels))

    train_losses, val_losses = run_cnn_experiments(dataset, runs=5)
    
    make_plot_losses_per_epochs(train_losses)
