import numpy as np
import cv2
import os

# full directory to the Images
directory = "/home/a1mb0t/Documents/Images/Originals/Resized/"
# directory = "/home/harry/Documents/Test_Sets/"

# List of Images and Labels sorted.
images, labels = sorted(os.listdir(directory + "/Images")), sorted(os.listdir(directory + "/Labels"))

# Empty lists that will contain the npy array equivalents of the pictures
npyImages, npyLabels = [], []

if len(images) == len(labels): # Validation of the size of the images and labels
  # for loop that manage the list of images and labels
  for i, l, z in zip(images, labels, range(len(images))):
    if ".jpeg" or ".JPEG" in i and l: # Validation of the extensions of the images and labels
      # print the name of the images and labels for check the correlation of them
      print(str(i) + " num: " + str(z) + "\n" + str(l) + " num: " + str(z) + "\n\n")

      npyImages.append(cv2.imread(directory + 'Images/' + str(i)))
      npyLabels.append(cv2.imread(directory + 'Labels/' + str(l), cv2.IMREAD_GRAYSCALE).reshape(500 * 500))
  # Conversion of the lists of images to numpy arrays
  npyImages = np.array(npyImages)
  npyLabels = np.array(npyLabels)

  # saving the arrays as .npy to the file system
  np.save(directory + 'npyImages', npyImages)
  np.save(directory + 'npyLabels', npyLabels)

  print("\n\tShape of npyImages: " + str(npyImages.shape))
  print("\n\tShape of npyLabels: " + str(npyLabels.shape))
else:
  # message printed if the length of the images and length are not the same
  print("The length of the directories is not the same!")
  print("\n\nImages length: " + str(len(images)) + "\nLabels length: " + str(len(labels)))
