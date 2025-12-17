from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import  SVC
import numpy as np
import random
from datetime import datetime
import sys
import os
import joblib as jl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_full_training_test_data, get_train_test_data_size

def rejection_mechanism_decision(max_decision_value, threshold):
    """
    as discussed in report, the higher the max decision function value,
    the far the sample is from the decision boundary, thus accepted.
    oppoisite for rejection.
    """
    return max_decision_value < threshold

def main():
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','data','cache'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    X_cached = os.path.join(cache_dir, 'X_svm.npy')
    y_cached =  os.path.join(cache_dir, 'y_svm.npy')
    label_map_cached = os.path.join(cache_dir, 'label_map.joblib')

    random.seed(42)
    np.random.seed(42)

    if(os.path.exists(X_cached) and os.path.exists(y_cached) and os.path.exists(label_map_cached)):
        X = np.load(X_cached)
        y = np.load(y_cached)
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
        jl.dump(label_map, label_map_cached)
    # 64% train, 16% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    

    scaler = StandardScaler()
    norm = Normalizer()

    X_train = scaler.fit_transform(X_train)
    X_train = norm.fit_transform(X_train)

    X_val = scaler.transform(X_val)
    X_val = norm.transform(X_val)

    X_test = scaler.transform(X_test)
    X_test = norm.transform(X_test)

    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    

    val_decision_values = svm_model.decision_function(X_val)

    val_max_decisions = np.max(val_decision_values, axis=1)
    

    thresholds = [10, 15, 20, 25, 30, 35, 40, 45]
    best_threshold = None
    best_score = -1
    
    print("\n=== choosing best threshold ===")
    for thresh in thresholds:
        threshold = np.percentile(val_max_decisions, thresh)
        
        # Evaluate on validation set
        is_rejected = val_max_decisions < threshold
        is_kept = ~is_rejected
        
        if not np.sum(is_kept):
            continue
            
        y_val_kept = y_val[is_kept]
        val_decision_kept = val_decision_values[is_kept]
        y_val_pred = svm_model.predict(X_val[is_kept])
        
        val_kept_acc = accuracy_score(y_val_kept, y_val_pred)
        rejection_rate = np.sum(is_rejected) / len(y_val)
        
        # Balanced score: prioritize kept accuracy but penalize high rejection
        if rejection_rate < 0.20:
            score = val_kept_acc
        else:
            score = val_kept_acc * (1 - rejection_rate * 0.5)
        
        print(f"Percentile {thresh:2d} | Threshold: {threshold:7.4f} | "
              f"Rejection: {rejection_rate*100:5.2f}% | Kept Acc: {val_kept_acc:.4f} | Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print(f"\n Best Threshold is: {best_threshold:.4f}\n")
    
    test_decision_values = svm_model.decision_function(X_test)
    test_max_decisions = np.max(test_decision_values, axis=1)
    
    is_rejected = rejection_mechanism_decision(test_max_decisions, best_threshold)
    is_kept = ~is_rejected
    
    # rejected marked as -1 (wrong)
    y_pred_all = svm_model.predict(X_test)
    y_pred_all[is_rejected] = -1  
    overall_accuracy = accuracy_score(y_test, y_pred_all)
    
    # Kept accuracy 
    if np.sum(is_kept) > 0:
        y_test_kept = y_test[is_kept]
        y_pred_kept = svm_model.predict(X_test[is_kept])
        kept_accuracy = accuracy_score(y_test_kept, y_pred_kept)
        class_report = classification_report(y_test_kept, y_pred_kept)
        conf_matrix = confusion_matrix(y_test_kept, y_pred_kept)
    else:
        kept_accuracy = 0.0
        class_report = "No samples kept"
        conf_matrix = np.array([[]])
    
    rejection_rate = np.sum(is_rejected) / len(y_test)
    
    print(f"\n{'='*60}")
    print(f"SVM with Rejection Mechanism (RBF kernel, C=10.0)")
    print(f"{'='*60}")
    print(f"Overall Accuracy:  {overall_accuracy:.4f} (rejected samples counted as wrong)")
    print(f"Rejection Rate:    {rejection_rate*100:.2f}%")
    print(f"Kept Accuracy:     {kept_accuracy:.4f} (only non-rejected samples)")
    print(f"Threshold:         {best_threshold:.4f}")
    print(f"{'='*60}\n")
    
    # save model and related components
    saved_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models'))
    os.makedirs(saved_models_dir, exist_ok=True)
    
    # Save model, scaler, normalizer, threshold, and label_map
    jl.dump(svm_model, os.path.join(saved_models_dir, 'svm_model.joblib'))
    jl.dump(scaler, os.path.join(saved_models_dir, 'svm_scaler.joblib'))
    jl.dump(norm, os.path.join(saved_models_dir, 'svm_normalizer.joblib'))
    jl.dump(best_threshold, os.path.join(saved_models_dir, 'svm_rejection_threshold.joblib'))
    jl.dump(label_map, os.path.join(saved_models_dir, 'svm_label_map.joblib'))
    print(f" Models saved to: {saved_models_dir}")
    
    # results and loggings
    base_results = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
    accuracy_folder = os.path.join(base_results, 'accuracy-plots')
    confusion_folder = os.path.join(base_results, 'confusion-matrices')
    os.makedirs(accuracy_folder, exist_ok=True)
    os.makedirs(confusion_folder, exist_ok=True)
    
    logs_file = os.path.join(base_results, 'logs.txt')
    accuracy_file = os.path.join(accuracy_folder, 'svm_rejection_accuracies.csv')
    
    with open(logs_file, 'a') as log_file:
        log_file.write(f"\n=== Run at {datetime.now()} ===\n")
        log_file.write("Changes/Approach: SVM with Decision Function Rejection Mechanism\n")
        log_file.write(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}\n")
        log_file.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        log_file.write(f"Rejection Rate: {rejection_rate*100:.2f}%\n")
        log_file.write(f"Kept Accuracy: {kept_accuracy:.4f}\n")
        log_file.write(f"Threshold: {best_threshold:.4f}\n")
    
    # more detailed accuracy csv
    if not os.path.exists(accuracy_file):
        with open(accuracy_file, 'w') as acc_file:
            acc_file.write("Timestamp,Overall_Acc,Rejection_Rate,Kept_Acc,Threshold\n")
    
    with open(accuracy_file, 'a') as acc_file:
        acc_file.write(f"{datetime.now()},{overall_accuracy:.4f},{rejection_rate:.4f},{kept_accuracy:.4f},{best_threshold:.4f}\n")

    with open(os.path.join(confusion_folder, 'svm_rejection_confusion_matrix.txt'), 'w') as cm_file:
        cm_file.write("Confusion Matrix (Kept Samples Only):\n")
        cm_file.write(np.array2string(conf_matrix) + "\n")
    
    classification_report_file = os.path.join(base_results, 'classification-reports', 'svm_rejection_classification_report.txt')
    os.makedirs(os.path.join(base_results, 'classification-reports'), exist_ok=True)
    with open(classification_report_file, 'w') as cr_file:
        cr_file.write("Classification Report (Kept Samples Only):\n")
        cr_file.write(class_report + "\n")



if __name__ == "__main__":
    main()