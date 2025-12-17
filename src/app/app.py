import cv2
import joblib as jl
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.feature_extraction import FeatureExtractor
import numpy as np


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_scaler = jl.load(os.path.join(base_dir, 'saved_models', 'svm_scaler.joblib'))
    model_normalizer = jl.load(os.path.join(base_dir, 'saved_models', 'svm_normalizer.joblib'))
    #svm is chosen here
    better_model = jl.load(os.path.join(base_dir, 'saved_models', 'svm_model.joblib'))
    threshold = jl.load(os.path.join(base_dir, 'saved_models', 'svm_rejection_threshold.joblib'))
    label_map = jl.load(os.path.join(base_dir, 'saved_models', 'svm_label_map.joblib'))

    featureExtractor = FeatureExtractor()

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True: 
        is_valid, frame = cam.read()
        if not is_valid:
            print("Error: Could not read frame from camera.")
            return
        

        processed = cv2.resize(frame, (64, 64))
        hist = featureExtractor.extract_color_histogram(processed)
        hist = hist/np.linalg.norm(hist)
        hog = featureExtractor.extract_hog(processed)
        hog = hog/np.linalg.norm(hog)
        feature_vector = np.concatenate((hist, hog))
        feature_vector = feature_vector/np.linalg.norm(feature_vector)
        scaled_features = model_scaler.transform([feature_vector])
        normalized_features = model_normalizer.transform(scaled_features)
        distances = better_model.decision_function(normalized_features)
        max_confidence = np.max(distances)
        is_rejected = max_confidence < threshold
        if is_rejected:
            result = "Prediction: Unknown Object"
        else:
            pred = better_model.predict(normalized_features)[0]
            renamed_label_map = {val:key for key, val in label_map.items()}
            predicted_class = renamed_label_map[pred]
            if predicted_class is None:
                result = "Prediction: Unknown Object"
            else:
                result = f"Prediction: {predicted_class} (Confidence: {max_confidence:.4f})"
        
        green, red = (0, 255, 0), (0, 0, 255)
        color = red if is_rejected else green
        cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Camera opened, press 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


    
    cam.release()
    cv2.destroyAllWindows()
    

    pass




if __name__ == "__main__":
    main()








