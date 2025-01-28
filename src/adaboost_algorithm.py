from sklearn.ensemble import AdaBoostClassifier
from functions import preprocess_images_from_dataset, perform_pca, train_model


if __name__ == "__main__":
    labels = ["0", "1", "2", "3", "4", "5", "6", "7"]

    X_train, y_train, X_test, y_test = preprocess_images_from_dataset(labels)
    X_train_pca, X_test_pca = perform_pca(X_train, X_test)
    
    ada_model = AdaBoostClassifier(algorithm="SAMME", n_estimators=15, learning_rate=1.0)
    
    _, _, models, avg_train_accuracy, avg_test_accuracy = train_model(ada_model, 5, X_train_pca, y_train, X_test_pca, y_test)

    print(f"Average train accuracy: {avg_train_accuracy}%")
    print(f"Average test accuracy: {avg_test_accuracy}%")

    