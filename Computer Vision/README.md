# Computer Vision Bootcamp 

A collection of three computer vision projects progressing from classical image processing to deep learning-based classification.

---

## 📁 Folder Structure

```
computer-vision-bootcamp/
│
├── README.md
│
├── Day1_Pencil_Sketch/
│   ├── Day1_Tasks
│   │   ├──CV_Image_Processing
│   ├── Day1_Assignment
│   │   ├── pencil_sketch.py
│   │   ├── test_images/
│   │   │   ├── test1.jpg
│   │   │   ├── test2.jpg
│   │   │   └── test3.jpg
│   │   ├── output_sketches/
│   │   │   ├── sketch1.jpg
│   │   │   ├── sketch2.jpg
│   │   │   └── sketch3.jpg
│   │   └── README.txt
│
├── Day2_Circle_Detector/
│   ├── circle_detector.py
│   ├── test_images/
│   │   ├── test1.jpg
│   │   ├── test2.jpg
│   │   └── test3.jpg
│   ├── results/
│   │   ├── result1.jpg
│   │   ├── result2.jpg
│   │   ├── result3.jpg
│   │   └── statistics.txt
│   └── README.txt
│
└── Day3_Cat_Dog_Classifier/
    ├── train.py
    ├── evaluate.py
    ├── data/
    │   ├── train/
    │   │   ├── cats/
    │   │   └── dogs/
    │   ├── val/
    │   │   ├── cats/
    │   │   └── dogs/
    │   └── test/
    │       ├── cats/
    │       └── dogs/
    ├── best_model.pth
    ├── training_curves.png
    ├── confusion_matrix.png
    └── README.txt

Computer Vision
project-root/
│
├── README.md
│
├── Day1/
│   ├── Day1_Assignment/
│   │   ├── pencil_sketch.py
│   │   ├── README.md
│   │   ├── output_sketches/
│   │   │   ├── colour_sketch1.jpg
│   │   │   ├── colour_sketch2.jpg
│   │   │   ├── colour_sketch3.jpg
│   │   │   ├── sketch1.jpg
│   │   │   ├── sketch2.jpg
│   │   │   └── sketch3.jpg
│   │   └── test_images/
│   │       ├── test1.jpg
│   │       ├── test2.jpg
│   │       └── test3.jpg
│   │
│   └── Day1_Tasks/
│       ├── CV_Image_Processing/
│       │   └── image_processing_pipeline.py
│       └── CV_Pixel_Histogram/
│           └── pixel_histogram_analysis.py
│
├── Day2/
│   ├── Day2_Assignment/
│   │   ├── circle_detector.py
│   │   ├── README.md
│   │   ├── results/
│   │   │   ├── result1.jpg
│   │   │   ├── result2.jpg
│   │   │   ├── result3.jpg
│   │   │   └── statistics.txt
│   │   └── test_images/
│   │       ├── test1.jpg
│   │       ├── test2.png
│   │       └── test3.jpg
│   │
│   └── Day2_Tasks/
│       ├── CV_Augmentation_Pipeline/
│       │   ├── augmentation_pipeline.py
│       │   ├── images/
│       │   │   └── test1.jpg
│       │   └── outputs/
│       │       ├── test1.jpg_output0.jpg
│       │       ├── test1.jpg_output1.jpg
│       │       ├── test1.jpg_output2.jpg
│       │       ├── test1.jpg_output3.jpg
│       │       └── test1.jpg_output4.jpg
│       │
│       └── CV_Feature_Matching/
│           ├── feature_matching.py
│           ├── view1.jpeg
│           └── view2.jpeg
│
└── Day3/
    ├── best_model.pth
    ├── confusion_matrix.png
    ├── evaluate.py
    ├── prediction_visualisation.png
    ├── README.md
    ├── train.py
    └── training_curves.png

```

---

## 🚀 What I built in my tenure till now:

### 1. Pencil Sketch Effect
Transforms photographs into realistic pencil sketch drawings using the dodge and burn technique. Implements grayscale conversion, image inversion, Gaussian blur, and division blending to create artistic pencil-like effects.

### 2. Circle Detector
Robust circle detection using the Hough Circle Transform. Identifies, analyzes, and visualizes circular objects in images with configurable parameters. Outputs annotated images with detected circles and detailed statistics (count, radius distribution, coordinates).

### 3. Cat vs Dog Classifier
Binary image classifier using transfer learning with PyTorch and ResNet18. Leverages pre-trained ImageNet weights, implements data augmentation, learning rate scheduling, and achieves >90% accuracy on test data. Includes training visualization and confusion matrix analysis.

---

## 🛠️ Tech Stack

**Libraries:**
- OpenCV - Image processing and computer vision
- NumPy - Numerical operations
- Matplotlib - Visualization
- PyTorch & Torchvision - Deep learning (Day 3)
- Scikit-learn & Seaborn - ML utilities and plotting (Day 3)

**Techniques:**
- Classical image processing (filtering, blending, transformations)
- Feature detection (Hough Transform, edge detection)
- Transfer learning and fine-tuning
- Data augmentation and normalization
- Model evaluation and metrics

---

## ⚙️ Setup

Install dependencies:

```bash
# For Day 1 & 2
pip install opencv-python numpy matplotlib

# For Day 3 (add these)
pip install torch torchvision scikit-learn seaborn
```

Or install everything:

```bash
pip install opencv-python numpy matplotlib torch torchvision scikit-learn seaborn
```

---

## 🎯 Quick Start

```bash
# Pencil Sketch
cd Day1_Pencil_Sketch
python pencil_sketch.py

# Circle Detection
cd Day2_Circle_Detector
python circle_detector.py

# Cat vs Dog Classifier
cd Day3_Cat_Dog_Classifier
python train.py        # Train model
python evaluate.py     # Evaluate on test set
```

---

## 📊 Key Features

**Day 1 - Pencil Sketch:**
- Side-by-side original and sketch visualization
- Adjustable blur kernel for different effects
- Error handling for invalid images
- Saves high-quality output sketches

**Day 2 - Circle Detector:**
- Configurable Hough Transform parameters
- Visual annotations with circle IDs and radii
- Statistical analysis (min/max/average radius)
- Handles overlapping and varying-sized circles

**Day 3 - Cat vs Dog Classifier:**
- ResNet18 with frozen backbone
- 5+ data augmentation techniques
- Learning rate scheduling (ReduceLROnPlateau)
- Saves best model checkpoints
- Generates training curves and confusion matrix
- Achieves 90-95% test accuracy

---

## 💡 Useful Tips

**General:**
- OpenCV uses BGR format, convert to RGB for matplotlib display
- Use `try-except` blocks for robust file handling
- Visualize intermediate steps during debugging

**Circle Detection:**
- Start with `param2=30`, lower it if circles are missed
- Increase `param2` if too many false positives
- Adjust `minDist` based on expected circle spacing

**Deep Learning:**
- Use Google Colab for free GPU access
- Monitor validation metrics, not training metrics
- Save checkpoints frequently to avoid losing progress
- Reduce batch size if running out of memory

---

## 📄 License

Educational project for Computer Vision Bootcamp.
