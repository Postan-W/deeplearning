import sys

import cv2
import numpy as np

# Load the input image -- 'chair.jpg'
# Convert it to grayscale
input_file = 'chair.jpg'
img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(img.shape)