import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    '''
    Preprocesses image
    Parameters:
        image_path: Path to original image
    Returns:
        image: Loaded image
        blur: Blurred image (gaussian blur)
    '''
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    return (image, blur)

def detect_circles(img):
    '''
    Detects circles in preprocessed image
    Parameters:
        img: Preprocessed image
    Returns:
        circles: information on circles
    '''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=50, minRadius=10, maxRadius=500)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def show_circles(image, circles, save_path=None):
    '''
    Draws detected circles on the image, prints statistics and saves result
    Parameters:
        image: Loaded image
        circles: Information on circles
        save_path: Path to save result
    '''
    if circles is not None:
        for circle in circles[0, :]:
            cv2.circle(image, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
            cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    text = (f"No of circles: {circles.shape[1]}\n"f"Average radius: {np.mean(circles[0, :, 2]):.2f}\n")
    plt.text(0.7, 0.7, text, fontsize=9, transform=plt.gcf().transFigure)
    plt.show()

def main():
    image_path = input("Enter path to image: ")
    save_path = input("Enter path to save result: ")
    image, blur = preprocess_image(image_path)
    circles = detect_circles(blur)
    show_circles(image, circles, save_path)

if __name__ == '__main__':
    main()
