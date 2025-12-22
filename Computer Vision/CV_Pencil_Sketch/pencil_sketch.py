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
        print("Invalid image path")
        return (None,None)
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
    
    if not os.path.isfile(image_path):
        print(f"Path is not a file: {image_path}")

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

    image, gray = pencil_sketch(image_path, blur_kernel)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = (s*0.7).astype(np.uint8)

    sketch = cv2.merge([h, s, gray])

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


def video_to_sketch(input_path, output_path, blur_kernel=21):
    '''
    Breaks video down into frames, converts each frame into a pencil sketch and remakes the video
    Parameters:
      input_path: original video path
      output_path: generated video path
      blur_kernel: size of kernel for gaussian blur
    '''
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open video")
        return
    
    if output_path is None:
        print(f"Invalid output path")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    print("Processing\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (blur_kernel, blur_kernel), 0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        
        out.write(sketch)
        print("▇", end="")

    out.release()
    cap.release()
    print(f"\nSaved sketch to: {output_path}")


def main():
    print("1. Pencil Sketch \n2. Colour Pencil Sketch \n3. Video Sketch")
    type = int(input("Choose function: "))

    blur_kernel = int(input("Enter kernel (odd number): "))
    if blur_kernel%2==0:
        print("Enter an odd number")

    else:

        if type == 1:
            image_path = input("Enter path of original image: ")
            save_path = input("Enter path to save output sketch: ")
            image, sketch = pencil_sketch(image_path, blur_kernel)
            if image is not None and sketch is not None:
                display_result(image, sketch, save_path)
            else:
                print("Enter a valid image path")

        elif type == 2:
            image_path = input("Enter path of original image: ")
            save_path = input("Enter path to save output sketch: ")
            image, sketch = colour_sketch(image_path, blur_kernel)
            if image is not None and sketch is not None:
                display_result_colour(image, sketch, save_path)
            else:
                print("Enter a valid image path")

        elif type == 3:
            input_path = input("Enter path to video: ")
            output_path = input("Enter path to save video sketch: ")
            video_to_sketch(input_path, output_path, blur_kernel)
        else:
            print("Enter a valid choice")


if __name__ == '__main__':
    main()
