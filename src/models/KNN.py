from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import numpy as np
import random
import sys
import os
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_full_training_test_data, get_train_test_data_size

def main():
    random.seed(42)
    (train_data, test_data) = get_full_training_test_data()
    X, y = [], []
    label_map = {}
    label_counter = 0

    for label, features in train_data.items():
        if label not in label_map:
            label_map[label] = label_counter
            label_counter += 1
        for feature in features:
            X.append(feature)
            y.append(label_map[label])


    for label, features in test_data.items():
        for feature in features:
            X.append(feature)
            y.append(label_map[label])

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    norm = Normalizer()
    X_train = norm.fit_transform(X_train)
    X_test = norm.transform(X_test)
    
    # results and loggings
    base_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    accuracy_folder = os.path.join(base_results, 'accuracy-plots')
    confusion_folder = os.path.join(base_results, 'confusion-matrices')
    os.makedirs(accuracy_folder, exist_ok=True)
    os.makedirs(confusion_folder, exist_ok=True)
    logs_file = os.path.join(base_results, 'logs.txt')
    accuracy_file = os.path.join(accuracy_folder, 'accuracies.csv')

    
    with open(accuracy_file, 'w') as f:
        f.write("timestamp,k,accuracy\n")

    

    K_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, int(np.sqrt(len(X_train)))]
    best_k = 1
    best_accuracy = 0.0
    with open(accuracy_file, 'w', newline='') as f:
        f.write(f"Timestamp,K,Accuracy\n")
    for k in K_values:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric='cosine', \
                                            weights='distance')
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Save confusion matrices per K
        cm_file = os.path.join(confusion_folder, f'confusion_matrix_k{k}.txt')
        with open(cm_file, 'a') as f:
            f.write(f"{datetime.now()} - Confusion Matrix K={k}:\n{conf_matrix}\n\n")

        # Write accuracy to CSV
        with open(accuracy_file, 'a') as f:
            
            f.write(f"{datetime.now()},{k},{accuracy:.4f}\n")

        print(f"K={k}: Accuracy={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    print(f"Best K: {best_k} with Accuracy: {best_accuracy:.4f}")
    with open(logs_file, 'a') as log_file:
        log_file.write(f"{datetime.now()} - Best K={best_k}, Accuracy={best_accuracy:.4f}\n")




if __name__ == "__main__":
    main()


