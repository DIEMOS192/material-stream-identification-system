import os
import cv2
import random
import numpy as np
from sys import path as sys_path
sys_path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FeatureExtractor:
    
    def flatten_image(self, image):
        image = cv2.resize(image, (64, 64))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.flatten()
    
    def extract_color_histogram(self):
        pass

    def extract_hog(self):
        pass

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
            extractor = FeatureExtractor()
            data_from = os.path.basename(data_folder)

            count = 1
            for  image in images:
                flattened = extractor.flatten_image(image)
                print(f"Category: {category}, Image {count}, Flattened size: {flattened.shape}")
                np.save(os.path.join(category_save_folder, f"{category}_{data_from}_flattened_{count}.npy"), flattened)
                count += 1





if __name__ == "__main__":
    main()




