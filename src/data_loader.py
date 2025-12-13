import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, target_size=(64, 64)):
    """
    Args:
        folder (str): Path to the folder containing class subfolders (cardboard, glass, etc.)
        target_size (tuple): Image size to resize to.
    Returns:
        images (np.array): Array of image data.
        labels (np.array): Array of corresponding labels.
    """
    # Member 1: Implement this
    pass

def extract_features(images, method='flatten'):
    """
    Args:
        images (np.array): Raw image data.
        method (str): 'flatten', 'histogram', etc.
    Returns:
        features (np.array): Processed features ready for training.
    """
    # Member 1: Implement this
    pass
