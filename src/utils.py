import random
import cv2
import os
import numpy as np
def read_random_image(directory):
    """Reads a random image from the specified directory."""
    images = [img for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        raise ValueError("No images found in the specified directory.")
    random_image = random.choice(images)
    image_path = os.path.join(directory, random_image)
    image = cv2.imread(image_path)
    return image

def calculate_training_data_size(training_data_folder=None):
    
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    if training_data_folder is None or training_data_folder.strip() == ''\
    or training_data_folder == 'raw':
        #calculate the whole training data size in this case
        total = 0
        for category in os.listdir(parent_dir):
            category_path =  os.path.join(parent_dir, category)
            if os.path.isdir(category_path):
                total += len([img for img in os.listdir(category_path) if img.endswith(('.png', '.jpg', '.jpeg'))])
        return total
    else:
        #calc number of images in a specific category 
        # ...number of images in paper or plastic or any specific...
        category_path = os.path.join(parent_dir, training_data_folder)
        if not os.path.isdir(category_path):
            raise ValueError(f"Provided category is: '{training_data_folder}' no such category, \
                             please either leave empty for total or specify a valid category.")
        total = len([img for img in os.listdir(category_path) if img.endswith(('.png', '.jpg', '.jpeg'))])
        return total
 


def get_full_training_test_data()->dict:
    features_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',\
                                        'features'))
    full_data = {}
    for data_folder in sorted(os.listdir(features_dir)):
        data_folder_path = os.path.join(features_dir, data_folder)
        if os.path.isdir(data_folder_path):
            folder_data = []
            for data_file in os.listdir(data_folder_path):
                if data_file.endswith('.npy'):
                    file_path = os.path.join(data_folder_path, data_file)
                    data_array = np.load(file_path)
                    folder_data.append(data_array)
            full_data[data_folder] = folder_data
    return full_data
        

def get_train_test_data_size()->tuple[int, int]:
    
    (train_data, test_data) = get_full_training_test_data()
    train_size = sum(len(v) for v in train_data.values())
    test_size = sum(len(v) for v in test_data.values())
    return train_size, test_size
            

def clear_outdated_train_test_data():
    features_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',\
                                       'features'))
    for data_folder in os.listdir(features_dir):
        data_folder_path = os.path.join(features_dir, data_folder)
        if os.path.isdir(data_folder_path):
            for data_file in os.listdir(data_folder_path):
                if data_file.endswith('.npy'):
                    file_path = os.path.join(data_folder_path, data_file)
                    os.remove(file_path)
    # clear cache
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cache'))
    if os.path.exists(cache_dir):
        for cache_file in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, cache_file)
            if os.path.isfile(file_path):
                os.remove(file_path)
     
def clear_aug():
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
def are_all_training_data_jpg()->bool:
    res:bool = True
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                       '..', 'data', 'raw'))
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if not img.lower().endswith('.jpg'):
                    res = False
                    return res
    return res


def get_max_data_folder()->int:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    max_size = 0
    for category in os.listdir(parent_dir):
        category_size = calculate_training_data_size(category)
        if category_size > max_size:
            max_size = category_size
    return max_size


if __name__ == "__main__":
    print(are_all_training_data_jpg())
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    for category in sorted(os.listdir(parent_dir)):
        category_path = os.path.join(parent_dir, category)
        if os.path.isdir(category_path):
            size = calculate_training_data_size(category)
            print(f"Category: {category}, Number of images: {size}")
    print("Max data folder size:", get_max_data_folder())
    clear_outdated_train_test_data()
    clear_aug()
