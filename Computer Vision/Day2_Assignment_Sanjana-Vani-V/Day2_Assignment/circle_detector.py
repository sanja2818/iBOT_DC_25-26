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

def detect_circles(img, dp=1, minDist=40, param1=50, param2=50, minRadius=30, maxRadius=200):
    '''
    Detects circles in preprocessed image
    Parameters:
        img: Preprocessed image
        dp: Inverse accumulator resolution ratio
        minDist: Minimum distance between centres
        param1: Upper canny threshold
        param2: Accumulator threshold
        minRadius: Minimum circle radius
        maxRadius: Maximum circle radius
    Returns:
        circles: information on circles
    '''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    return circles

def visualise_circles(image, circles, save_path=None):
    '''
    Draws detected circles on the image, prints statistics and saves result
    Parameters:
        image: Loaded image
        circles: Information on circles
        save_path: Path to save result
    '''
    if circles is not None:
        for circle in circles[0, :]:
            if circle[2] <50:
                cv2.circle(image, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)
                cv2.circle(image, (circle[0], circle[1]), 2, (255, 0, 0), 3)
            elif circle[2] <100:
                cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(image, (circle[0], circle[1]), 2, (0, 255, 0), 3)
            else:
                cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
                cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    text1 = (f"No of circles: {circles.shape[1]}\n"f"Average radius: {np.mean(circles[0, :, 2]):.2f}\n")
    text2 = (f"BLUE: SMALL \nGREEN: MEDIUM \nRED: LARGE")
    plt.text(0.7, 0.7, text1, fontsize=9, transform=plt.gcf().transFigure)
    plt.text(0.2, 0.2, text2, fontsize=9, transform=plt.gcf().transFigure)
    plt.show()

def calculate_statistics(circles, save_path):
    '''
    Calculates statistics and saves the result
    Parameters:
        circles: Information on circles
        save_path: Path to save result
    '''

    with open(r'.\results\statistics.txt', 'a') as f:
        text1 = (f"No of circles: {circles.shape[1]}\n"f"Average radius: {np.mean(circles[0, :, 2]):.2f}\n")
        f.write(save_path)
        f.write('\n')
        f.write(text1)

        for circle in circles[0, :]:
            f.write(f'Coordinates: ({circle[0]},{circle[1]}) \t Radius: {circle[2]}\n')
        f.write('\n')


def main():
    image_path = input("Enter path to image: ")
    save_path = input("Enter path to save result: ")
    image, blur = preprocess_image(image_path)
    circles = detect_circles(blur)
    visualise_circles(image, circles, save_path)
    calculate_statistics(circles, save_path)

if __name__ == '__main__':
    main()
