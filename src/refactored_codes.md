
refacored augmentation code

# def fixed_augment_cycle(image):
#     geoAug = GeometricAugmentor()
#     photoAug = PhotometricAugmentor()
#     image = geoAug.rotate(image)
#     if random.random() < 0.5:
#         image = geoAug.flip(image)
#     if random.random() < 0.3:
#         image = photoAug.adjust_brightness_contrast(image)
#     image = photoAug.Gauss_blur(image)
#     return image

# def adjust_brightness_contrast(self, image):
    
#     brightness = random.uniform(-30, 30)
#     contrast = random.uniform(0.9, 1.1)
#     img = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
#     return img

# def main_v2():
#     root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#     raw_data_folder = os.path.join(root, 'data', 'raw')
#     aug_data_folder = os.path.join(root, 'data', 'augmented')
#     for category in os.listdir(raw_data_folder):
#         full_size = calculate_training_data_size(os.path.join(raw_data_folder, category))
#         max_size = get_max_data_folder()
#         aug_size = int((max_size - full_size)) + 20
#         if aug_size <= 0:
#             aug_size = 20
#         category_path = os.path.join(raw_data_folder, category)
#         save_category_path = os.path.join(aug_data_folder, category)
#         if not os.path.exists(save_category_path):
#             os.makedirs(save_category_path)
#         imgs = os.listdir(category_path)
#         count = 1
#         attempts = 0
#         while count <= aug_size:
#             attempts += 1
#             if attempts > full_size * 2:
#                 break

#             img = random.choice(imgs)
#             img_name = 'augmented_' + category + '_' + str(count) + '.jpg'
#             image = cv2.imread(os.path.join(category_path, img))
#             if image is None:
#                 continue
#             for i in range(2):
#                 augmented_image = fixed_augment_cycle(image.copy())
#                 if augmented_image is None:
#                     continue
#                 img_name = 'augmented_' + category + '_' + str(count) + '.jpg'
#                 cv2.imwrite(os.path.join(save_category_path, img_name), augmented_image)
#                 count += 1







feature extraction that reached 0.8619

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
        pixel_bins = [8, 8, 8]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, pixel_bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(gray, pixels_per_cell=(16, 16),\
                        cells_per_block=(2, 2),
                        orientations=9,
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