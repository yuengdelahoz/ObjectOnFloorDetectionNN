import numpy as np
import cv2
import os

directory = "datasets/training_data/images/"
images = []
for file in os.listdir(directory):
    if file.endswith(".jpg") or file.endswith(".JPG"):
        print(str(file))
        image = cv2.imread(directory+'/'+str(file))
        # print(image.shape)
        images.append(image)
images = np.array(images)
np.save('images',images)
# img2 = np.load('images.npy')
# print(img2.shape)
# cv2.imshow('img',img2[0])
# cv2.waitKey()

directory = "datasets/training_data/labels/"
images = []
for file in os.listdir(directory):
    if file.endswith(".jpg") or file.endswith(".JPG"):
        print(str(file))
        image = cv2.imread(directory+'/'+str(file),cv2.IMREAD_GRAYSCALE)
        rows,cols = image.shape
        image = image.reshape(rows*cols)
        images.append(image)
images = np.array(images)
np.save('labels',images)
img2 = np.load('labels.npy')
# print(img2.shape)
img1 = img2[0].reshape(536, 858)
cv2.imshow('img',img1)
cv2.waitKey()
