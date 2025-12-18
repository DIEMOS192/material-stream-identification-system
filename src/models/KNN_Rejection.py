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
from utils import get_full_training_test_data
import joblib as jl




def main():
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','data','cache'))
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    X_cached = os.path.join(cache_dir, 'X_knn.npy')
    y_cached =  os.path.join(cache_dir, 'y_knn.npy')
    label_map_cached = os.path.join(cache_dir, 'knn_label_map.joblib')

    random.seed(42)
    np.random.seed(42)

    if(os.path.exists(X_cached) and os.path.exists(y_cached) and os.path.exists(label_map_cached)):
        X = np.load(X_cached)
        y =  np.load(y_cached)
        label_map = jl.load(label_map_cached)
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
        X = np.array(X)
        y = np.array(y)
        np.save(X_cached, X)
        np.save(y_cached, y)
        assert X.shape[0] == y.shape[0], "Mismatch between number of samples and labels"
        jl.dump(label_map, os.path.join(cache_dir, 'knn_label_map.joblib'))

    # 64% train, 16% validation, 20% test
    X_temp, X_test, y_temp, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

    norm = Normalizer()
    X_train = norm.fit_transform(X_train)
    X_val = norm.transform(X_val)
    X_test = norm.transform(X_test)

    jl.dump(norm, os.path.join(save_dir, 'knn_normalizer.joblib'))
    jl.dump(label_map, os.path.join(save_dir, 'knn_label_map.joblib'))

    K_values = [1, 3, 4, 5, 7, 9]
    best_k = 1
    best_accuracy = 0.0
    
    # results and loggings
    base_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    accuracy_folder = os.path.join(base_results, 'accuracy-plots')
    confusion_folder = os.path.join(base_results, 'confusion-matrices')
    classification_folder = os.path.join(base_results, 'classification-reports')
    os.makedirs(accuracy_folder, exist_ok=True)
    os.makedirs(confusion_folder, exist_ok=True)
    os.makedirs(classification_folder, exist_ok=True)
    logs_file = os.path.join(base_results, 'logs.txt')
    accuracy_file = os.path.join(accuracy_folder, 'knn_accuracies.csv')

    
    with open(logs_file, 'a') as log_file:
        log_file.write("Algorithm: K-Nearest Neighbors (Distance-Based Rejection)\n")
        log_file.write("K-collection: " + str(set(K_values)) + "\n")
        log_file.write(f"\n=== Run at {datetime.now()} ===\n")
        log_file.write("Changes/Approach: Manhattan distance rejection with validation threshold\n")
        log_file.write(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")
        log_file.write("K collection results:\n")

    with open(classification_folder + '/knn_classification_report.txt', 'a') as class_file:
        class_file.write(f"\n=== Run at {datetime.now()} ===\n")
        class_file.write("Distance-based rejection mechanism (for negative class detection)\n")




    

   
    with open(accuracy_file, 'w', newline='') as f:
        f.write(f"Timestamp,K,Overall_Acc,Rejection_Rate,Kept_Acc,Threshold\n")
    
    for k in K_values:
        knn_model = KNeighborsClassifier(n_neighbors=k, metric='manhattan', weights='distance')
        knn_model.fit(X_train, y_train)
        
        val_distances, degenerates = knn_model.kneighbors(X_val)
        val_mean_distances = []
        for dist in val_distances:
            val_mean_distances.append(sum(dist) / k)
        val_mean_distances = np.array(val_mean_distances)
        
        thresholds = np.percentile(val_mean_distances, [50, 60, 70, 75, 80, 85, 90, 95])
        best_threshold = None
        best_val_score = 0.0
        
        y_val_pred = knn_model.predict(X_val)
        
        for threshold in thresholds:
            is_rejected = np.array([val > threshold for val in val_mean_distances])
            is_kept = np.array([not r for r in is_rejected])
            if True not in is_kept:
                continue

            y_val_kept =[]
            y_val_pred_kept = []
            for i in range(len(y_val)):
                if is_kept[i]:
                    y_val_kept.append(y_val[i])
                    y_val_pred_kept.append(y_val_pred[i])
            
            kept_acc = accuracy_score(y_val_kept, y_val_pred_kept)
            rejection_rate = 1 - sum(is_kept) / len(y_val)
            
            # setting a rejection limit for reasonable evaluation
            if rejection_rate < 0.2:
                score = kept_acc
            else:
                score = kept_acc * (1 - rejection_rate * 0.5) 
            
            if score > best_val_score:
                best_val_score = score
                best_threshold = threshold
        
        if best_threshold is None:
            best_threshold = np.percentile(val_mean_distances, 90)
        
        
        test_distances, degenerates = knn_model.kneighbors(X_test)

        test_mean_distances = []
        for dist in test_distances:
            test_mean_distances.append(sum(dist) / k)
        test_mean_distances = np.array(test_mean_distances)

        test_is_kept = np.array([val <= best_threshold for val in test_mean_distances])
        
        y_pred = knn_model.predict(X_test)
        overall_accuracy = accuracy_score(y_test, y_pred)
        rejection_rate = 1 - sum(test_is_kept) / len(y_test)
        
        if True in test_is_kept:
            y_test_kept = []
            y_pred_kept = []
            for i in range(len(y_test)):
                if test_is_kept[i]:
                    y_test_kept.append(y_test[i])
                    y_pred_kept.append(y_pred[i])
            kept_accuracy = accuracy_score(y_test_kept, y_pred_kept)
        else:
            kept_accuracy = 0.0
        
        # save model and rejection threshold
        jl.dump(knn_model, os.path.join(save_dir, f'knn_model_k{k}.joblib'))
        jl.dump(best_threshold, os.path.join(save_dir, f'knn_rejection_threshold_k{k}.joblib'))
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Save detailed results
        cm_file = os.path.join(confusion_folder, f'confusion_matrix_k{k}.txt')
        with open(cm_file, 'w') as f:
            f.write(f"{datetime.now()} - K={k}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
            f.write(f"Rejection Rate: {rejection_rate:.2%}\n")
            f.write(f"Accuracy on Kept Samples: {kept_accuracy:.4f}\n")
            f.write(f"Rejection Threshold (mean manhattan distance): {best_threshold:.4f}\n")
        
        with open(accuracy_file, 'a') as f:
            f.write(f"{datetime.now()},{k},{overall_accuracy:.4f},{rejection_rate:.4f},{kept_accuracy:.4f},{best_threshold:.4f}\n")
        
        with open(logs_file, 'a') as log_file:
            log_file.write(f"K={k}: Acc={overall_accuracy:.4f}, Reject={rejection_rate:.2%}, Kept_Acc={kept_accuracy:.4f}, Threshold={best_threshold:.4f}\n")
        
        print(f"K={k}: Acc={overall_accuracy:.4f}, Reject={rejection_rate:.2%}, Kept_Acc={kept_accuracy:.4f}, Threshold={best_threshold:.4f}")
        
        if overall_accuracy > best_accuracy:
            best_k = k
            best_accuracy = overall_accuracy

    print(f"Best K: {best_k} with Accuracy: {best_accuracy:.4f}")
    with open(logs_file, 'a') as log_file:
        log_file.write(f"{datetime.now()} - Best K={best_k}, Accuracy={best_accuracy:.4f}\n")




if __name__ == "__main__":
    main()