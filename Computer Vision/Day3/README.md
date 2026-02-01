Name:
Sanjana Vani V

Running the program:
- Run train.py to obtain best_model.pth (saved weights after training) and Training and Validation curves
- Run evaluate.py to display information from testing

Final Test Accuracy:
Test Accuracy: 97.80%

Data Augmentation Techniques Used:
- Resize to 256 pixels
- Random crop to 224 × 224
- Random horizontal flip
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization using ImageNet mean and standard deviation

Learning Rate Schedule Used:
- Adam optimizer with an initial learning rate of 0.001
- ReduceLROnPlateau scheduler
  - Monitors validation loss
  - Reduces learning rate by a factor of 0.5
  - Patience of 3 epochs

Any Challenges Faced:
- I was initially unable to use my GPU to as I kept getting torch.cuda.is_available()=False
  I attempted to use colab but I had the same issue and due to lack of error management while using device = 'cuda' the code ran for hours instead of minutes
  I had to run the code on kaggle instead after debugging
- I had some difficulty selecting a good learning rate

Bonus Assignments:
Bonus 3 - Visualizing Predictions
- Created a grid of 
    - 5 correct cat predictions
    - 5 correct dog predictions
    - 5 cats predicted as dogs
    - 5 dogs predicted as cats

Dependancies:
- python3
- pytorch with cuda
- numpy
- matplotlib
- seaborn
- sklearn
