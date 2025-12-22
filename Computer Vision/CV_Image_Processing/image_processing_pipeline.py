import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def image_processing(image_path):
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    
    if not os.path.isfile(image_path):
        print(f"Path is not a file: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Enter a valid image path")
    
    #gaussian blur 
    blurred = cv2.GaussianBlur(image, (7,7), 0)

    #binary thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    #canny edge detection
    edge = cv2.Canny(image, 50, 150)

    #plotting
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    images = [image, blurred, binary, edge]
    titles = ["Original", "Gaussian Blur", "Threshold", "Canny"]
    for a, img, title in zip(axes.flat, images, titles):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        a.imshow(rgb)
        a.set_title(title)
        a.axis('off')
    plt.tight_layout()
    plt.show()

image_path = input("Enter path to image: ")
image_processing(image_path)