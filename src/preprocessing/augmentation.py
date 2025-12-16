
import cv2
import random
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import  calculate_training_data_size, get_max_data_folder


class GeometricAugmentor:

    def rotate(self,image, angle=15):
        angle = random.uniform(-angle, angle)

        (height, width, degenerate) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), \
                                 borderMode=cv2.BORDER_REFLECT_101)
        return rotated
        
        
    def flip(self, image, hr = 0.5): #horizontal only
        return cv2.flip(image, 1) if random.random() < hr else image
        
    def scale(self, image, scaling=None):#0.9-1.1x
        if scaling is None or scaling < 0.9 or scaling > 1.1:
            scaling = random.uniform(0.9, 1.1)
        (height, width, channels) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 0, scaling)
        scaled = cv2.warpAffine(image, rotation_matrix, (width, height))
        return scaled
    # def random_crop(self, image, pos:str=None, size:float=None):
    #     if pos is None or pos not in ['tl', 'tr', 'bl', 'br', 'c']:
    #         pos = random.choice(['tl', 'tr', 'bl', 'br', 'c'])
    #     if size is None or size < 0.5 or size > 0.9:
    #         size = random.uniform(0.5, 0.9)
    #     new_img = None
    #     (height, width, channels) = image.shape
    #     new_height = int(height * size)
    #     new_width = int(width * size)
    #     if pos == 'tl':
    #         new_img = image[0:new_height, 0:new_width]
    #     elif pos == 'tr':
    #         new_img = image[0:new_height, width - new_width:width]
    #     elif pos == 'bl':
    #         new_img = image[height - new_height:height, 0:new_width]
    #     elif pos == 'br':
    #         new_img = image[height - new_height:height, width - new_width:width]
    #     elif pos == 'c':
    #         start_y = (height - new_height) // 2
    #         start_x = (width - new_width) // 2
    #         new_img = image[start_y:start_y + new_height, start_x:start_x + new_width]

    #     return new_img
    def random_resize(self, image, factor:float=None):
        if factor is None or factor < 0.8 or factor > 1.2:
            factor = random.uniform(0.8, 1.2)
        (height, width, channels) = image.shape
        resized = cv2.resize(image, (int(width * factor), int(height * factor)))
        return resized

class PhotometricAugmentor:

    def adjust_brightness(self, image, constant = 30): # 30 or -30
        if constant not in [-30, 30]:
            constant = random.choice([-30, 30])
        img = cv2.convertScaleAbs(image, alpha=1, beta=constant)
        return img
    def adjust_contrast(self, image, constant = None): # between 0.8 and 1.2
        if constant is None or constant < 0.8 or constant > 1.2:
            constant = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(image, alpha=constant, beta=0)
        return img
    
    def adjust_brightness_contrast(self, image):
        
        brightness = random.uniform(-30, 30)
        contrast = random.uniform(0.9, 1.1)
        img = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return img
    # def add_noise(self, image, noise = None): # from 5 to 15 # should we keep it?
    #     if noise is None or noise < 5 or noise > 15:
    #         noise = random.randint(5, 15)
    #     gauss = np.random.normal(0, noise, image.shape)
    #     noisy = image.astype(np.float32) + gauss
    #     noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    #     return noisy
    def Gauss_blur(self, image, ksize = 3, prob = 0.2):
        return cv2.GaussianBlur(image, (ksize, ksize), 0) \
            if random.random() < prob else image 


def choose_random_augmentation():


    types = [GeometricAugmentor, PhotometricAugmentor]
    AugmentorClass = random.choice(types)
    algorithm = AugmentorClass()
    GeoMethods = ['rotate', 'flip', 'scale', 'random_resize']
    PhotoMethods = ['adjust_brightness', 'adjust_contrast', 'add_noise']
    if AugmentorClass == GeometricAugmentor:
        method_name = random.choice(GeoMethods)
    else:
        method_name = random.choice(PhotoMethods)
    method = getattr(algorithm, method_name)
    return method



def fixed_augment_cycle(image):
    geoAug = GeometricAugmentor()
    photoAug = PhotometricAugmentor()
    image = geoAug.rotate(image)
    if random.random() < 0.5:
        image = geoAug.flip(image)
    if random.random() < 0.3:
        image = photoAug.adjust_brightness_contrast(image)
    image = photoAug.Gauss_blur(image)
    return image

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

def main_v2():
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
        while count <= aug_size:
            attempts += 1
            if attempts > full_size * 2:
                break

            img = random.choice(imgs)
            img_name = 'augmented_' + category + '_' + str(count) + '.jpg'
            image = cv2.imread(os.path.join(category_path, img))
            if image is None:
                continue
            for i in range(2):
                augmented_image = fixed_augment_cycle(image.copy())
                if augmented_image is None:
                    continue
                img_name = 'augmented_' + category + '_' + str(count) + '.jpg'
                cv2.imwrite(os.path.join(save_category_path, img_name), augmented_image)
                count += 1   

if __name__ == "__main__":
    # main()
    main_v2()

