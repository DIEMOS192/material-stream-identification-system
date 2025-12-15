import os
import cv2
import numpy as np
from sys import path as sys_path
from skimage.feature import hog
sys_path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FeatureExtractor:
    
    def flatten_image(self, image):
        image = cv2.resize(image, (64, 64))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.flatten()
    
    def extract_color_histogram(self, image, bins=(8, 8, 8)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            bins, [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(3, 3),
            block_norm="L2-Hys"
        )
        return hog_features
    
    def extract_features(self, image):
        image = cv2.resize(image, (128, 128))

        color_hist = self.extract_color_histogram(image)
        hog_feat = self.extract_hog(image)

        return np.hstack([color_hist, hog_feat])

def read_folder_images(folder):
    images = []
    for img_name in os.listdir(folder):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
    return images

def main():
    material_dir = os.listdir("dataset")
    save_folder = "features"

    os.makedirs(save_folder, exist_ok=True)
    extractor = FeatureExtractor()

    for category in material_dir:
        category_save_folder = os.path.join(save_folder, category)
        os.makedirs(category_save_folder, exist_ok=True)

        category_path = os.path.join("dataset", category)

        images = read_folder_images(category_path)
        extractor = FeatureExtractor()
        data_from = os.path.basename("dataset")

        count = 1
        for  image in images:
            flattened = extractor.extract_features(image)
            print(f"Category: {category}, Image {count}, Flattened size: {flattened.shape}")
            np.save(os.path.join(category_save_folder, f"{category}_{data_from}_features_{count}.npy"), flattened)
            count += 1





if __name__ == "__main__":
    main()