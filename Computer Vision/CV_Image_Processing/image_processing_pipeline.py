import numpy as np
import cv2
import os

def image_processing(image_path):
    if image_path is None:
        print(f"Enter a valid image path")
    
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    
    if not os.path.isfile(image_path):
        print(f"Path is not a file: {image_path}")
    