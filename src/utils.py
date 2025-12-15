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
 


def get_full_training_test_data():
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',\
                                        'features'))
    training_data = {}
    test_data = {}
    train_portion = 0.8
    random.seed(42)
    for data_folder in os.listdir(dir):
        data_folder_path = os.path.join(dir, data_folder)
        if os.path.isdir(data_folder_path):
            folder_data = []
            for data_file in os.listdir(data_folder_path):
                if data_file.endswith('.npy'):
                    file_path = os.path.join(data_folder_path, data_file)
                    data_array = np.load(file_path)
                    folder_data.append(data_array)
            random.shuffle(folder_data)
            for_train = int(len(folder_data) * train_portion)
            training_data[data_folder] = folder_data[:for_train]
            test_data[data_folder] = folder_data[for_train:]
    return training_data, test_data
        

def get_train_test_data_size()->tuple[int, int]:
    
    (train_data, test_data) = get_full_training_test_data()
    train_size = sum(len(v) for v in train_data.values())
    test_size = sum(len(v) for v in test_data.values())
    return train_size, test_size
            

def clear_outdated_train_test_data():
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data',\
                                       'features'))
    for data_folder in os.listdir(dir):
        data_folder_path = os.path.join(dir, data_folder)
        if os.path.isdir(data_folder_path):
            for data_file in os.listdir(data_folder_path):
                if data_file.endswith('.npy'):
                    file_path = os.path.join(data_folder_path, data_file)
                    os.remove(file_path)
     

def are_all_training_data_jpg()->bool:
    res:bool = True
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    for folder in os.listdir(dir):
        folder_path = os.path.join(dir, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                if not img.lower().endswith('.jpg'):
                    res = False
                    return res
    return res

            


if __name__ == "__main__":
    # print(are_all_training_data_jpg())
    pass