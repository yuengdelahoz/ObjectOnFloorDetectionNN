import cv2
import numpy as np
# This file will show an specific image of the labels and image
cv2.imshow('image', np.load("/home/harry/Documents/Test/Images/Resized/npyImages.npy")[38])
cv2.imshow('image2', np.load("/home/harry/Documents/Test/Images/Resized/npyLabels.npy")[38].reshape(500, 500))
cv2.waitKey()

# img = np.array(cv2.LoadImageM("/home/harry/Documents/Test/Images/obj/4x3/File000.jpeg"))

# print(img.shape)