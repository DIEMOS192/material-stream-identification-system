import random
import cv2
import os
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
 

