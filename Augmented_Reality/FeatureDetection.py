import cv2
import numpy as np

# Getting the Image ready for feature detection
input_image = cv2.imread('Augmented-Reality-from-scratch-master/book.jpg')
# Resize the image to a specific width and height
resize_width = 800
resize_height = 600
input_image = cv2.resize(input_image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Initiate ORB object
orb = cv2.ORB_create(nfeatures=1000)

# Find the keypoints with ORB
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# Draw only the location of the keypoints without size
final_keypoints = cv2.drawKeypoints(gray_image, keypoints, input_image, (0, 255, 0))

# Display the result
cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
