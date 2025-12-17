from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import  SVC
import numpy as np
import random
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_full_training_test_data, get_train_test_data_size
import joblib as jl

def main():
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','data','cache'))
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','saved_models'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    X_cached = os.path.join(cache_dir, 'X_svm.npy')
    y_cached =  os.path.join(cache_dir, 'y_svm.npy')


    random.seed(42)
    np.random.seed(42)

    if(os.path.exists(X_cached) and os.path.exists(y_cached)):
        X = np.load(X_cached)
        y =  np.load(y_cached)
        label_map = jl.load(os.path.join(cache_dir, 'svm_label_map.joblib'))
    else:
        full_data = get_full_training_test_data()
        X, y = [], []
        label_map = {}
        label_counter = 0
        for label, features in full_data.items():
            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1
            for feature in features:
                X.append(feature)
                y.append(label_map[label])
        
        jl.dump(label_map, os.path.join(cache_dir, 'svm_label_map.joblib'))

        X = np.array(X)
        y = np.array(y)
        np.save(X_cached, X)
        np.save(y_cached, y)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    norm = Normalizer()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = norm.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test = norm.transform(X_test)

    jl.dump(scaler, os.path.join(save_dir, 'svm_scaler.joblib'))
    jl.dump(norm, os.path.join(save_dir, 'svm_normalizer.joblib'))
    jl.dump(label_map, os.path.join(save_dir, 'svm_label_map.joblib'))

    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    jl.dump(svm_model, os.path.join(save_dir, 'svm_model.joblib'))
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.4f}")
    # results and loggings
    base_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    accuracy_folder = os.path.join(base_results, 'accuracy-plots')
    confusion_folder = os.path.join(base_results, 'confusion-matrices')
    os.makedirs(accuracy_folder, exist_ok=True)
    os.makedirs(confusion_folder, exist_ok=True)
    logs_file = os.path.join(base_results, 'logs.txt')
    accuracy_file = os.path.join(accuracy_folder, 'svm_accuracies.csv')
    with open(logs_file, 'a') as log_file:
        log_file.write(f"\n=== Run at {datetime.now()} ===\n")
        log_file.write("Changes/Approach: kernel SVM\n")
        log_file.write(f"Number of training samples: {len(X_train)}, Number of test samples: {len(X_test)}\n")
        log_file.write(f"Accuracy: {accuracy:.4f}\n")
    with open(accuracy_file, 'a') as acc_file:
        acc_file.write(f"{datetime.now()},{accuracy:.4f}\n")

    with open(os.path.join(confusion_folder, 'svm_confusion_matrix.txt'), 'w') as cm_file:
        cm_file.write("Confusion Matrix:\n")
        cm_file.write(np.array2string(conf_matrix) + "\n")
    
    classification_report_file = os.path.join(base_results, 'classification-reports', 'svm_classification_report.txt')
    with open(classification_report_file, 'w') as cr_file:
        cr_file.write("Classification Report:\n")
        cr_file.write(class_report + "\n")



if __name__ == "__main__":
    main()