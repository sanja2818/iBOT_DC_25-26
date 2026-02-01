Running the program:
- Run pencil_sketch.py
- Choose one of the following options:
  - 1: Generate a grayscale pencil sketch from an image
  - 2: Generate a colour pencil sketch from an image
  - 3: Convert an entire video into a pencil sketch video
- Enter an odd-valued Gaussian blur kernel size
- Enter valid input and output file paths

Output:
- Image modes (1 & 2):
  - Displays the original image and the generated sketch side by side.
  - Saves the sketch image to the specified output path.
- Video mode (3):
  - Processes the video frame by frame.
  - Saves a grayscale pencil-sketch-style video to the specified output path.

Image Processing Techniques Used:
- Conversion from BGR to grayscale
- Image inversion
- Gaussian blur for edge softening
- Image division for pencil sketch effect
- HSV colour space manipulation for colour sketch
  - Reduced saturation to simulate coloured pencil shading

Video Processing Pipeline:
- Video is decomposed into individual frames
- Each frame is converted into a grayscale pencil sketch
- Frames are recombined into a video

Any Challenges Faced:
- Ensuring correct color space conversions between opencv and matplotlib
- Choosing parameters for proper colour pencil sketch effect
- Preventing crashes from invalid file paths or unsupported formats

Bonus Assignments:
- Bonus 1: Adjustable Blur Parameter
    - User input for blur kernel size
- Bonus 2: Colour Pencil Sketch
    - Converts to HSV colour space
    - Slight desaturation to be realistic
- Bonus 3: Video Processing
    - Processes video frame by frame and returns gray pencil sketch video

Dependencies:
- python3
- cv2
- numpy
- matplotlib
- os
