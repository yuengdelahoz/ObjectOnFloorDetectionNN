import numpy as np
import cv2
import os

directory = "/home/harry/Documents/Training_Sets/"

images, labels = sorted(os.listdir(directory + "/Images")), sorted(os.listdir(directory + "/Labels"))

npyImages, npyLabels = [], []

if len(images) == len(labels):
  for i, l in zip(images, labels):
    if ".jpeg" or ".JPEG" in i and l:
      print(str(i) + "\n" + str(l) + "\n\n")

      npyImages.append(cv2.imread(directory + 'Images/' + str(i)))
      npyLabels.append(cv2.imread(directory + 'Labels/' + str(l), cv2.IMREAD_GRAYSCALE).reshape(500 * 500))

  npyImages = np.array(npyImages)
  npyLabels = np.array(npyLabels)

  np.save('npyImages', npyImages)
  np.save('npyLabels', npyLabels)
else:
  print("The length of the directories is not the same!")
