import numpy as np
import cv2
import os

def load_images(folder):
    images = dict()
    for filename in os.listdir(folder):
        path = os.path.join(folder,filename)
        img = cv2.imread(path)
        if img is not None:
            images[filename] = img
    return images

def augment_5(image):
    outputs = []

    height, width = image.shape[:2]
    center = (width//2, height//2)

    #1. random rotation
    random_angle = np.random.uniform(-30, 30)  # Random angle between -30 and 30
    rotation_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    outputs.append(rotated)

    #2. shearing
    shear_matrix = np.float32([[1, 0.2, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(image, shear_matrix, (width, height))
    outputs.append(sheared)

    #3. contrast
    image_f = image.astype(np.float32)
    contrast_factor = 1.5
    contrasted = np.clip(128 + contrast_factor * (image_f - 128), 0, 255).astype(np.uint8)
    outputs.append(contrasted)

    #4. saturated
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype('float32')
    saturation_scale = 1.5
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    saturated = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
    outputs.append(saturated)

    #5. hue shift
    hue_shift = np.random.uniform(30, 70)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hue_shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    outputs.append(hue_shifted)

    return outputs

def save_augmented(filename, outputs, folder):
    savename = filename + '_output'
    for i in range(len(outputs)):
        outputname = savename + f"{i}.jpg"
        cv2.imwrite(os.path.join(folder,outputname), outputs[i])

images = load_images(r".\images")
for filename, image in images.items():
    outputs = augment_5(image)
    save_augmented(filename, outputs, r".\outputs")
