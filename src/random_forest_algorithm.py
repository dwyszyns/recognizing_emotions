import numpy as np
import cv2
import glob
import dlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from functions import preprocess_images_from_dataset, perform_pca, train_model


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
    labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    X_train, y_train, X_test, y_test = preprocess_images_from_dataset(labels)
    X_train_pca, X_test_pca = perform_pca(X_train, X_test)

    rf = RandomForestClassifier(criterion='gini', max_depth=7, max_features='sqrt', min_samples_leaf=1, n_estimators=50)
    
    scores_train, scores_test, models, avg_train_accuracy, avg_test_accuracy = train_model(rf, 5, X_train_pca, y_train, X_test_pca, y_test)
    print(f"Average train accuracy: {avg_train_accuracy}")
    print(f"Average test accuracy: {avg_test_accuracy}")

    best_model_index = np.argmax(scores_test)
    best_model = models[best_model_index]
    
    show_confusion_matrix(best_model, X_test_pca, y_test)

    