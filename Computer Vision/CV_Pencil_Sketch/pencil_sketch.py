import numpy as np
import cv2
import matplotlib.pyplot as plt

def pencil_sketch(image_path, blur_kernel=21):
    '''
    Converts a colour image into a grayscale pencil sketch effect image
    Parameters:
      image_path (str): file path to colour image
      blur kernel (int): size of kernel for gaussian blur
    Returns:
      tuple containing original BGR image and generated grayscale sketch
    '''

    #reading the image
    image = cv2.imread(image_path)
    if image is None:
        return (None,None)

    #converting to grayscale and inverting
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray

    #blurring the image and inverting
    blurred = cv2.GaussianBlur(inverted, (blur_kernel,blur_kernel), 0)
    invblurred = 255 - blurred

    #dividing and scaling
    sketch = cv2.divide(gray, invblurred, scale=256)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    return (image, sketch)


def colour_sketch(image_path, blur_kernel=21):
    '''
    Converts a colour image into a colour pencil sketch effect image
    Parameters:
      image_path (str): file path to colour image
      blur kernel (int): size of kernel for gaussian blur
    Returns:
      tuple containing original BGR image and generated colour sketch (BGR)
    '''

    #reading the image
    image = cv2.imread(image_path)
    if image is None:
        return (None,None)

    #converting to hsv and inverting value
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_inv = 255 - v

    #blurring the value and inverting
    v_blur = cv2.GaussianBlur(v_inv, (blur_kernel, blur_kernel), 0)
    v_invblur = 255 - v_blur

    #dividing and scaling
    v_sketch = cv2.divide(v, v_invblur, scale=256)
    v_sketch = np.clip(v_sketch, 0, 255).astype(np.uint8)
    s = (s*0.7).astype(np.uint8)

    sketch = cv2.merge([h, s, v_sketch])

    final_sketch = cv2.cvtColor(sketch, cv2.COLOR_HSV2BGR)

    return (image, final_sketch)


def display_result (original, sketch, save_path=None):
    '''
    Displays original image and processed image side by side, and saves the new image
    Parameters:
      original: original image
      sketch: generated image (greyscale)
      save_path: path to save sketch
    '''

    #converting to rgv for matplotlib
    original_RGB = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_RGB)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(sketch, cmap='gray')
    axes[1].set_title("Pencil Sketch")
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        cv2.imwrite(save_path, sketch)
        print(f"Sketch saved to: {save_path}")

    plt.show()


def display_result_colour (original, sketch, save_path=None):
    '''
    Displays original image and processed image side by side, and saves the new image
    Parameters:
      original: original image
      sketch: generated image (BGR)
      save_path: path to save sketch
    '''

    #converting to rgv for matplotlib
    original_RGB = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    sketch_RGB = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_RGB)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(sketch_RGB)
    axes[1].set_title("Pencil Sketch")
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        cv2.imwrite(save_path, sketch)
        print(f"Sketch saved to: {save_path}")

    plt.show()


def main():
    print("1. Pencil Sketch \n2. Colour Pencil Sketch")
    type = int(input("Choose function: "))

    image_path = input("Enter path of original image: ")
    blur_kernel = int(input("Enter kernel (odd number): "))
    if blur_kernel%2==0:
        print("Enter an odd number")

    else:
        save_path = input("Enter path to save output sketch: ")

        if type == 1:
            image, sketch = pencil_sketch(image_path, blur_kernel)
            if image is not None and sketch is not None:
                display_result(image, sketch, save_path)
            else:
                print("Enter a valid image path")
        elif type == 2:
            image, sketch = colour_sketch(image_path, blur_kernel)
            if image is not None and sketch is not None:
                display_result_colour(image, sketch, save_path)
            else:
                print("Enter a valid image path")
        else:
            print("Enter a valid choice")


if __name__ == '__main__':
    main()