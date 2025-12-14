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

