import numpy as np
import cv2

image = cv2.imread("/home/harry/Desktop/File_001.jpeg")

(h, w) = image.shape[:2]
center = (w / 2, h / 2)
 
# rotate the image by 180 degrees
M = cv2.getRotationMatrix2D(center, 0, 1.0)
rotated = cv2.warpAffine(image, M, (h, w))
# cv2.imshow("rotated", rotated)
# cv2.waitKey(0)

cv2.imwrite('270.jpeg', image)