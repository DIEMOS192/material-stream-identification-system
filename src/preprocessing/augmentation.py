
import cv2
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import read_random_image


class GeometricAugmentor:

    def rotate(self, angle, image):#small degree (±10-25°)

        if angle < -15 or angle > 15:
            angle = random.randint(-15, 15)
            (height, width, channels) = image.shape
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
            return rotated
        
        
    def flip(self, image):
        degree = random.choice([90, 180, 270])
        (height, width, channels) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
        flipped = cv2.warpAffine(image, rotation_matrix, (width, height))
        return flipped
        
    def scale(self, image):#0.9-1.1x
        scaling = random.uniform(0.9, 1.1)
        (height, width, channels) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 0, scaling)
        scaled = cv2.warpAffine(image, rotation_matrix, (width, height))
        return scaled
    def random_crop(self, image, pos=None, size=None):
        if pos is None or pos not in ['tl', 'tr', 'bl', 'br', 'c']:
            pos = random.choice(['tl', 'tr', 'bl', 'br', 'c'])
        if size is None:
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
    def random_resize(self, image):
        new_s = random.uniform(0.8, 1.2)
        (height, width, channels) = image.shape
        resized = cv2.resize(image, (int(width * new_s), int(height * new_s)))
        return resized

class PhotometricAugmentor:

    def adjust_brightness(self):
        pass
    def adjust_contrast(self):
        pass
    def slight_color_jitter(self):
        pass
    def adjust_hue(self):
        pass
    def add_noise(self):#Gaussian, (low variance)
        pass


def choose_random_augmentation(image):
    pass

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    row_data_folder = os.path.join(project_root, 'data', 'raw')
    save_data_folder = os.path.join(project_root, 'data', 'augmented')
    path = os.path.join(row_data_folder, 'plastic')

    image = read_random_image(path)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    resized = GeometricAugmentor().random_resize(image)
    cv2.imshow('Resized Image', resized)
    cv2.waitKey(0)
    writePath = os.path.join(save_data_folder, 'resized_image.jpg')
    cv2.imwrite(writePath, resized)
    # rotated = GeometricAugmentor().rotate(20, image)
    # cv2.imshow('Rotated Image', rotated)
    # cv2.waitKey(0)
    # writePath = '../data/augmented/rotated_image.jpg'
    # cv2.imwrite(writePath, rotated)

