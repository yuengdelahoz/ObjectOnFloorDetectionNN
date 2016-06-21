import cv2

cv2.imshow('image', np.load('npyImages.npy')[101])
cv2.imshow('image2', np.load('npyLabels.npy')[101].reshape(500, 500))
cv2.waitKey()