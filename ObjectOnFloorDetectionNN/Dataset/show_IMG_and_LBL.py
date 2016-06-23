import cv2
import numpy as np
# This file will show an specific image of the labels and image
cv2.imshow('image', np.load('/home/harry/Documents/Training_Sets/npyImages.npy')[1061])
cv2.imshow('image2', np.load('/home/harry/Documents/Test_Sets/npyLabels.npy')[1061].reshape(500, 500))
cv2.waitKey()