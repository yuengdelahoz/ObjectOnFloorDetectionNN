import numpy as np
import cv2
import os

# full directory to the Images
directory = "/home/a1mb0t/Documents/Images/Labaled/Resized/"

# String that contains the saving file path
savingDirectory = "/media/a1mb0t/1CF7-3E37/"
# directory = "/home/harry/Documents/Test_Sets/"

# List of Images and Labels sorted.
images, labels = sorted(os.listdir(directory + "/Images")), sorted(os.listdir(directory + "/Labels"))

# Empty lists that will contain the npy array equivalents of the pictures
npyTrainingImages, npyTrainingLabels = [], []
npyTestingImages, npyTestingLabels = [], []

if len(images) == len(labels): # Validation of the size of the images and labels
  # for loop that manage the list of images, labels and iteration number
  for i, l, z in zip(images, labels, range(len(images))):
    if ".jpeg" or ".JPEG" in i and l: # Validation of the extensions of the images and labels
      if z < 10000: # If that append the images and labels from iteration 0 - 9999 to the Training Lists
        # print the name of the images and labels for check the correlation of them
        print("Adding to training files: " + str(i) + " num: " + str(z) + "\n" + "Adding to training files: " + str(l) + " num: " + str(z) + "\n\n")

        # Appending the images to the training lists
        npyTrainingImages.append(cv2.imread(directory + 'Images/' + str(i)))
        npyTrainingLabels.append(cv2.imread(directory + 'Labels/' + str(l), cv2.IMREAD_GRAYSCALE).reshape(500 * 500))
      else: #else that will append the images and labels from iteration 1000 - (len(images) - 1) to the Testing Lists
        if z == 10000: # If that will save the training npy array into npy files
          # Printing the status of the code: Status of the npy array creation
          print("\n\tCreating the npy arrays of the Training sets...\n")

          npyTrainingImages = np.array(npyTrainingImages)
          print("npyTrainingImages npy array created!")

          npyTrainingLabels = np.array(npyTrainingLabels)
          print("npyTrainingLabels npy array created!")

          print("\n\tShape of npyTrainingImages: " + str(npyTrainingImages.shape))
          print("\n\tShape of npyTrainingLabels: " + str(npyTrainingLabels.shape))

          # saving the arrays of the Training sets as .npy to the file system
          print("\n\tSaving the npy files of the Training sets...\n")
          
          np.save(savingDirectory + 'npyTrainingImages', npyTrainingImages)
          np.save(savingDirectory + 'npyTrainingLabels', npyTrainingLabels)


        # print the name of the images and labels for check the correlation of them
        print("Adding to testing files: " + str(i) + " num: " + str(z) + "\n" + "Adding to testing files: " + str(l) + " num: " + str(z) + "\n\n")

        npyTestingImages.append(cv2.imread(directory + 'Images/' + str(i)))
        npyTestingLabels.append(cv2.imread(directory + 'Labels/' + str(l), cv2.IMREAD_GRAYSCALE).reshape(500 * 500))

  npyTestingImages = np.array(npyTestingImages)
  print("npyTestingImages npy array created!")

  npyTestingLabels = np.array(npyTestingLabels)
  print("npyTestingLabels npy array created!")
  
  print("\n\tShape of npyTestingImages: " + str(npyTestingImages.shape))
  print("\n\tShape of npyTestingLabels: " + str(npyTestingLabels.shape))

  # saving the arrays of the Training sets as .npy to the file system
  print("\n\tSaving the npy files of the Testing sets...\n")
  np.save(savingDirectory + 'npyTestingImages', npyTestingImages)
  
  np.save(savingDirectory + 'npyTestingLabels', npyTestingLabels)
else:
  # message printed if the length of the images and length are not the same
  print("The length of the directories is not the same!")
  print("\n\nImages length: " + str(len(images)) + "\nLabels length: " + str(len(labels)))

print("\n\n\tFinish!")