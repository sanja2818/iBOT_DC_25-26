import numpy as np
import cv2
import matplotlib.pyplot as plt

view1 = cv2.imread("view1.jpeg")
view2 = cv2.imread("view2.jpeg")

orb = cv2.ORB_create(nfeatures=500)

gray1 = cv2.cvtColor(view1, cv2.COLOR_BGR2GRAY)
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)     

gray2 = cv2.cvtColor(view2, cv2.COLOR_BGR2GRAY)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)     

matcher = cv2.BFMatcher()
matches = matcher.match(descriptors1, descriptors2)

final_img = cv2.drawMatches(view1, keypoints1, view2, keypoints2, matches[:50], None)
final_img = cv2.resize(final_img, (1000, 650))

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)) 
plt.title("Feature Matches")
plt.axis('off')  
plt.show()
     
