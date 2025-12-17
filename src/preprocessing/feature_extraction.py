import os
import cv2
import random
import numpy as np
from sys import path as sys_path

from sklearn.preprocessing import Normalizer
sys_path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import clear_outdated_train_test_data
from skimage.feature import hog


class FeatureExtractor:
    
    def flatten_image(self, image):
        image = cv2.resize(image, (64, 64))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.flatten()
    
    def extract_color_histogram(self, image):
        pixel_bins = [14, 12, 12]  # Changed from (8, 8, 8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, pixel_bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # changed from 16, 16 # main place to edit for balancing KNN nad svm
        features = hog(gray, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), # was 2, 2
                        orientations=9, # was 9
                        block_norm='L2-Hys',
                        visualize=False)
        return features
        

def read_folder_images(folder):
    images = []
    for img_name in os.listdir(folder):
        if img_name.endswith(('.jpg')):
            img_path = os.path.join(folder, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
    return images

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_data_folder = os.path.join(root, 'data', 'raw')
    aug_data_folder = os.path.join(root, 'data', 'augmented')
    save_folder = os.path.join(root, 'data', 'features')
    for data_folder in [raw_data_folder, aug_data_folder]:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for category in os.listdir(data_folder):
            category_save_folder = os.path.join(save_folder, category)
            if not os.path.exists(category_save_folder):
                os.makedirs(category_save_folder)
            category_path = os.path.join(data_folder, category)

            images = read_folder_images(category_path)
            data_from = os.path.basename(data_folder)

            count = 1
            for  image in images:
                image = cv2.resize(image, (64, 64))
                histogram = FeatureExtractor().extract_color_histogram(image)
                histogram = histogram/np.linalg.norm(histogram)

                hog = FeatureExtractor().extract_hog(image)
                hog = hog/np.linalg.norm(hog)

                histogram_hog = np.concatenate((histogram, hog))
                histogram_hog = histogram_hog/np.linalg.norm(histogram_hog)

                vector_length = histogram_hog.shape[0]
                np.save(os.path.join(category_save_folder, f"{category}_{data_from}_histogram_hog_{count}.npy"), histogram_hog)
                count += 1
    print("Feature extraction completed.")
    print("Vector length:", vector_length)





if __name__ == "__main__":
    clear_outdated_train_test_data()
    main()