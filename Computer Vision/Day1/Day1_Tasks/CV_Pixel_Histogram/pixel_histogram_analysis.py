import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

image_path = input("Enter path to image: ")

if not os.path.exists(image_path):
    print(f"Image file not found: {image_path}")
    
if not os.path.isfile(image_path):
    print(f"Path is not a file: {image_path}")

image = cv2.imread(image_path)
if image is None:
    print(f"Enter a valid image path")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pixels = gray.flatten()

mean = np.mean(pixels)
median = np.median(pixels)
std = np.std(pixels)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(gray, cmap = 'gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].hist(pixels, bins=256, range=(0, 255))
axes[1].set_title("Intensity Histogram")
axes[1].set_xlabel("Pixel Intensity")
axes[1].set_ylabel("Frequency")
    
stats_text = (f"Mean: {mean:.2f}\n"f"Median: {median:.2f}\n"f"Std Dev: {std:.2f}")

axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes, ha="right", va="top")

plt.tight_layout()
plt.show()
    