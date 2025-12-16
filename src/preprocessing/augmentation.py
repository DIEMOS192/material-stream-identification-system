
import cv2
import random
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import  calculate_training_data_size, get_max_data_folder


class GeometricAugmentor:

    def rotate(self,image, angle=None):# -15 to 15 degrees

        if angle is None or angle < -15 or angle > 15:
            angle = random.randint(-15, 15)
        (height, width, degenerate) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
        
        
    def flip(self, image, hr_vr=None): #horizontal or vertical flip
        if hr_vr is None or hr_vr not in [0, 1]:
            hr_vr = random.choice([0, 1])
        flipped = cv2.flip(image, hr_vr)
        return flipped
        
    def scale(self, image, scaling=None):#0.9-1.1x
        if scaling is None or scaling < 0.9 or scaling > 1.1:
            scaling = random.uniform(0.9, 1.1)
        (height, width, channels) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 0, scaling)
        scaled = cv2.warpAffine(image, rotation_matrix, (width, height))
        return scaled
    def random_crop(self, image, pos:str=None, size:float=None):
        if pos is None or pos not in ['tl', 'tr', 'bl', 'br', 'c']:
            pos = random.choice(['tl', 'tr', 'bl', 'br', 'c'])
        if size is None or size < 0.5 or size > 0.9:
            size = random.uniform(0.5, 0.9)
        new_img = None
        (height, width, channels) = image.shape
        new_height = int(height * size)
        new_width = int(width * size)
        if pos == 'tl':
            new_img = image[0:new_height, 0:new_width]
        elif pos == 'tr':
            new_img = image[0:new_height, width - new_width:width]
        elif pos == 'bl':
            new_img = image[height - new_height:height, 0:new_width]
        elif pos == 'br':
            new_img = image[height - new_height:height, width - new_width:width]
        elif pos == 'c':
            start_y = (height - new_height) // 2
            start_x = (width - new_width) // 2
            new_img = image[start_y:start_y + new_height, start_x:start_x + new_width]

        return new_img
    def random_resize(self, image, factor:float=None):
        if factor is None or factor < 0.8 or factor > 1.2:
            factor = random.uniform(0.8, 1.2)
        (height, width, channels) = image.shape
        resized = cv2.resize(image, (int(width * factor), int(height * factor)))
        return resized

class PhotometricAugmentor:

    def adjust_brightness(self, image, constant = None): #between 30 and -30
        if constant is None or constant not in range(-30, 31):
            constant = random.randint(-30, 30)
        img = cv2.convertScaleAbs(image, alpha=1, beta=constant)
        return img
    def adjust_contrast(self, image, constant = None): # between 0.8 and 1.2
        if constant is None or constant < 0.8 or constant > 1.2:
            constant = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(image, alpha=constant, beta=0)
        return img
    def add_noise(self, image, noise = None): # from 5 to 15 # should we keep it?
        if noise is None or noise < 5 or noise > 15:
            noise = random.randint(5, 15)
        gauss = np.random.normal(0, noise, image.shape)
        noisy = image.astype(np.float32) + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy


def choose_random_augmentation():
    types = [GeometricAugmentor, PhotometricAugmentor]
    AugmentorClass = random.choice(types)
    algorithm = AugmentorClass()
    GeoMethods = ['rotate', 'flip', 'scale', 'random_crop', 'random_resize']
    PhotoMethods = ['adjust_brightness', 'adjust_contrast', 'add_noise']
    if AugmentorClass == GeometricAugmentor:
        method_name = random.choice(GeoMethods)
    else:
        method_name = random.choice(PhotoMethods)
    method = getattr(algorithm, method_name)
    return method


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_data_folder = os.path.join(root, 'data', 'raw')
    aug_data_folder = os.path.join(root, 'data', 'augmented')
    for category in os.listdir(raw_data_folder):
        full_size = calculate_training_data_size(os.path.join(raw_data_folder, category))
        max_size = get_max_data_folder()
        aug_size = int((max_size - full_size)) + 20
        if aug_size <= 0:
            aug_size = 20
        category_path = os.path.join(raw_data_folder, category)
        save_category_path = os.path.join(aug_data_folder, category)
        if not os.path.exists(save_category_path):
            os.makedirs(save_category_path)
        imgs = os.listdir(category_path)
        count = 1
        attempts = 0
        while True:
            attempts += 1
            if attempts > calculate_training_data_size() * 2:
                break
            img = random.choice(imgs)
            img_name = 'augmented_' + category + '_' + str(count) + '.jpg'
            image = cv2.imread(os.path.join(category_path, img))
            if image is None:
                continue
            for i in range(random.randint(1,4)):
                method = choose_random_augmentation()
                image = method(image)
            augmented_image = image
            if augmented_image is None:
                continue
            cv2.imwrite(os.path.join(save_category_path, img_name), augmented_image)
            count += 1
            if count > aug_size:
                break

    

if __name__ == "__main__":
    main()
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # row_data_folder = os.path.join(project_root, 'data', 'raw')
    # save_data_folder = os.path.join(project_root, 'data', 'augmented')

    # image = read_random_image(os.path.join(row_data_folder, 'plastic'))
    # cv2.imshow('Original Image', image)
    # cv2.waitKey(0)
    # noisy = PhotometricAugmentor().add_noise(image, 10)
    # cv2.imshow('Noisy Image', noisy)
    # cv2.waitKey(0)
    # writePath = os.path.join(save_data_folder, 'noisy_image.jpg')
    # cv2.imwrite(writePath, noisy)
    # rotated = GeometricAugmentor().rotate(20, image)
    # cv2.imshow('Rotated Image', rotated)
    # cv2.waitKey(0)
    # writePath = '../data/augmented/rotated_image.jpg'
    # cv2.imwrite(writePath, rotated)

