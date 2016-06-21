import cv2
import numpy as np
# This file will show an specific image of the labels and image
cv2.imshow('image', np.load('npyImages.npy')[106])
cv2.imshow('image2', np.load('npyLabels.npy')[106].reshape(500, 500))
cv2.waitKey()