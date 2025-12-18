import cv2
import numpy as np
import joblib as jl
import os
import random
from skimage.feature import hog


class FeatureExtractor:


    
    def extract_color_histogram(self, image):
        pixel_bins = [14, 12, 12]  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, pixel_bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        features = hog(gray, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), 
                        orientations=9, 
                        block_norm='L2-Hys',
                        visualize=False)
        return features

class GeometricAugmentor:
    def rotate(self,image, angle=15):
        angle = random.uniform(-angle, angle)

        (height, width, degenerate) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), \
                                 borderMode=cv2.BORDER_REFLECT_101)
        return rotated
    def scale(self, image, scaling=None):#0.9-1.1x
        if scaling is None or scaling < 0.9 or scaling > 1.1:
            scaling = random.uniform(0.9, 1.1)
        (height, width, degenerate) = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 0, scaling)
        scaled = cv2.warpAffine(image, rotation_matrix, (width, height), \
                                borderMode=cv2.BORDER_REFLECT_101)
        return scaled
    def random_resize(self, image, factor:float=None):
        if factor is None or factor < 0.8 or factor > 1.2:
            factor = random.uniform(0.8, 1.2)
        (height, width, channels) = image.shape
        resized = cv2.resize(image, (int(width * factor), int(height * factor)))
        return resized
    

class PhotometricAugmentor:
    def adjust_brightness(self, image, constant = 30): 
        if constant not in [-30, 30]:
            constant = random.choice([-30, 30])
        img = cv2.convertScaleAbs(image, alpha=1, beta=constant)
        return img
    def adjust_contrast(self, image, constant = None): 
        if constant is None or constant < 0.8 or constant > 1.2:
            constant = random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(image, alpha=constant, beta=0)
        return img
    
    def add_noise(self, image, noise = None): 
        if noise is None or noise < 5 or noise > 15:
            noise = random.randint(5, 15)
        gauss = np.random.normal(0, noise, image.shape)
        noisy = image.astype(np.float32) + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    def Gauss_blur(self, image, ksize = 3, prob = 0.2):
        return cv2.GaussianBlur(image, (ksize, ksize), 0) \
            if random.random() < prob else image
class utils:
    def choose_random_augmentation(self):


        types = [GeometricAugmentor, PhotometricAugmentor]
        AugmentorClass = random.choice(types)
        algorithm = AugmentorClass()
        GeoMethods = ['rotate', 'scale','random_resize']
        PhotoMethods = ['adjust_brightness', 'adjust_contrast', 'Gauss_blur', 'add_noise']
        if AugmentorClass == GeometricAugmentor:
            method_name = random.choice(GeoMethods)
        else:
            method_name = random.choice(PhotoMethods)
        method = getattr(algorithm, method_name)
        return method
    
    def calculate_training_data_size(self, folder):
        total = 0
        for img_name in os.listdir(folder):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                total += 1
        return total
    def clear_aug(self):
        aug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'augmented'))
        if not os.path.isdir(aug_dir):
            return
        for category in os.listdir(aug_dir):
            category_path = os.path.join(aug_dir, category)
            if not os.path.isdir(category_path):
                continue

            for entry in os.listdir(category_path):
                entry_path = os.path.join(category_path, entry)
                if os.path.isfile(entry_path):
                    os.remove(entry_path)

    def Augmentor(self, dataFilePath):
        random.seed(42)
        raw_data_folder = dataFilePath
        aug_data_folder = dataFilePath + "_augment"
        if os.path.exists(aug_data_folder):
            for entry in os.listdir(aug_data_folder):
                entry_path = os.path.join(aug_data_folder, entry)
                if os.path.isfile(entry_path):
                    os.remove(entry_path)
        if not os.path.exists(aug_data_folder):
            os.makedirs(aug_data_folder)
        full_size = self.calculate_training_data_size(raw_data_folder)
        aug_size = full_size + 20
        imgs = os.listdir(raw_data_folder)
        imgs = [img for img in imgs if img.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(imgs)
        if not imgs:
            return aug_data_folder
        count = 1
        attempts = 0
        while count <= aug_size:
            attempts += 1
            if attempts > full_size * 2:
                break
            img = random.choice(imgs)
            img_name = 'augmented_' + str(count) + '.jpg'
            image = cv2.imread(os.path.join(raw_data_folder, img))
            if image is None:
                continue
            for i in range(random.randint(1,4)):
                method = self.choose_random_augmentation()
                image = method(image)
            augmented_image = image
            if augmented_image is None:
                continue
            cv2.imwrite(os.path.join(aug_data_folder, img_name), augmented_image)
            count += 1
        return aug_data_folder

    def clear_aug_folder(self, aug_data_folder):
        if os.path.exists(aug_data_folder):
            for entry in os.listdir(aug_data_folder):
                entry_path = os.path.join(aug_data_folder, entry)
                if os.path.isfile(entry_path):
                    os.remove(entry_path)
            os.rmdir(aug_data_folder)


def predict(dataFilePath, bestModelPath, add_augmentation: bool = False):

    # model loading
    best_model = jl.load(os.path.join(bestModelPath, 'svm_model.joblib'))
    scaler = jl.load(os.path.join(bestModelPath, 'svm_scaler.joblib'))
    normalizer = jl.load(os.path.join(bestModelPath, 'svm_normalizer.joblib'))
    threshold = jl.load(os.path.join(bestModelPath, 'svm_rejection_threshold.joblib'))
    label_map = jl.load(os.path.join(bestModelPath, 'svm_label_map.joblib'))

    util = utils()
    aug_data_folder = dataFilePath + "_augment"
    
    # cleanup previous run's augmented folder
    util.clear_aug_folder(aug_data_folder)
    
    # read images from original folder
    images = []
    for image_file in os.listdir(dataFilePath):
        if not image_file.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(dataFilePath, image_file)
        image = cv2.imread(img_path)
        images.append((image_file, image))

    # augmentation (optional)
    if add_augmentation:
        aug_data_folder = util.Augmentor(dataFilePath)
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
            predicted_class = "unknown"
        else:
            pred = best_model.predict(normalized)[0]
            renamed_label_map = {val:key for key, val in label_map.items()}
            predicted_class = renamed_label_map[pred]
            if predicted_class is None:
                predicted_class = "unknown"
        predictions.append(predicted_class)

    return predictions


def main(add_augmentation: bool = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataFilePath = os.path.join(script_dir, '..', 'data', 'samples')
    bestModelPath = os.path.join(script_dir, 'models', 'saved_models')
    
    predictions = predict(dataFilePath, bestModelPath, add_augmentation)
    print(f"Total predictions: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"  {i+1}: {pred}")




if __name__ == "__main__":
    add_augmentation:bool = False
    # please set it to true if you want additional data augmentation
    # as this will at least increase the sample size by 20 images
    main(add_augmentation =True)