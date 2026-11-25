import cv2
import numpy as np
import joblib as jl
import os
import random
import sys
from openpyxl import Workbook

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.feature_extraction import FeatureExtractor
from preprocessing.augmentation import choose_random_augmentation


def predict(dataFilePath, bestModelPath, add_augmentation: bool = False):

    # model loading
    best_model = jl.load(os.path.join(bestModelPath, 'svm_model.joblib'))
    scaler = jl.load(os.path.join(bestModelPath, 'svm_scaler.joblib'))
    normalizer = jl.load(os.path.join(bestModelPath, 'svm_normalizer.joblib'))
    threshold = jl.load(os.path.join(bestModelPath, 'svm_rejection_threshold.joblib'))
    label_map = jl.load(os.path.join(bestModelPath, 'svm_label_map.joblib'))
    inv_label_map = {val: key for key, val in label_map.items()}

    # read images from original folder
    images = []
    for image_file in os.listdir(dataFilePath):
        if not image_file.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(dataFilePath, image_file)
        image = cv2.imread(img_path)
        images.append((image_file, image))

    # augmentation (if enabled)
    if add_augmentation:
        random.seed(42)
        aug_data_folder = dataFilePath + "_augment"
        if os.path.exists(aug_data_folder):
            for entry in os.listdir(aug_data_folder):
                entry_path = os.path.join(aug_data_folder, entry)
                if os.path.isfile(entry_path):
                    os.remove(entry_path)
        else:
            os.makedirs(aug_data_folder)
        for image_file in os.listdir(dataFilePath):
            if not image_file.endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(dataFilePath, image_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            for i in range(random.randint(1, 4)):
                method = choose_random_augmentation()
                image = method(image)
            cv2.imwrite(os.path.join(aug_data_folder, f"aug_{image_file}"), image)
        for image_file in os.listdir(aug_data_folder):
            if not image_file.endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(aug_data_folder, image_file)
            image = cv2.imread(img_path)
            images.append((image_file, image))

    # feature extraction
    features = []
    for img_name, image in images:
        image = cv2.resize(image, (64, 64))
        feature_extractor = FeatureExtractor()
        hist = feature_extractor.extract_color_histogram(image)
        hist = hist/np.linalg.norm(hist)
        hog_feat = feature_extractor.extract_hog(image)
        hog_feat = hog_feat/np.linalg.norm(hog_feat)
        feature_vector = np.concatenate((hist, hog_feat))
        feature_vector = feature_vector/np.linalg.norm(feature_vector)
        features.append((img_name, feature_vector))

    # processing features
    predictions = []
    for img_name, feature in features:
        scaled = scaler.transform([feature])
        normalized = normalizer.transform(scaled)
        distances = best_model.decision_function(normalized)
        max_confidence = np.max(distances)
        is_rejected = []
        for dist in distances[0]:
            is_rejected.append(dist < threshold)
        if all(is_rejected):
            predicted_class = -1  # unknown/rejected
        else:
            pred = best_model.predict(normalized)[0]
            predicted_class = pred
        predicted_label = 'unknown' if predicted_class == -1 else inv_label_map.get(predicted_class, str(predicted_class))
        predictions.append((img_name, predicted_label))

    return predictions


def save_predictions_excel(results, output_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "predictions"
    ws.append(["ImageName", "predictedlabel"])
    for image_name, predicted_label in results:
        ws.append([image_name, predicted_label])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)
    wb.save(output_path)


def main(add_augmentation: bool = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataFilePath = os.path.join(script_dir, '..', 'data', 'testset-C')
    bestModelPath = os.path.join(script_dir, 'models', 'saved_models')
    
    results = predict(dataFilePath, bestModelPath, add_augmentation)
    output_file = os.path.join(script_dir, '..', 'results', 'predictions.xlsx')
    save_predictions_excel(results, output_file)
    print(f"Saved predictions to {output_file}")




if __name__ == "__main__":
    add_augmentation:bool = False
    # please set it to true if you want additional data augmentation
    # as this will at least increase the sample size by 20 images
    # test data folder is assumed to be in ../data and named 'samples'
    # adjust accordingly if different and the code should run fine

    # labels as IDs were stated in classroom.
    main(add_augmentation)