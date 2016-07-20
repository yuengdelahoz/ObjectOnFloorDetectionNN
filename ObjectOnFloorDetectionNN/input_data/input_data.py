import os, os.path
import sys
import numpy as np
import tensorflow as tf
import cv2
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

class Dataset:
	def __init__(self, images):
		self._images = images[0]
		self._labels = images[1]
		self.samples = self._images.shape[0]
		self.epoch_index = 0
		
	def next_batch(self, batch_size):
		if batch_size > self.samples:
			raise ValueError("Dataset error...batch size is greater than the number of samples")

		start = self.epoch_index
		self.epoch_index += batch_size

		if self.epoch_index > self.samples:
			# Shuffle the indexes
			temp = np.array([i for i in range(self.samples)])
			np.random.shuffle(temp)
			# Shuffle the data
			self._images = self._images[temp]
			self._labels = self._labels[temp]
			# Start again with new shuffle data
			start = 0
			self.epoch_index = batch_size

		end = self.epoch_index

		return self._images[start:end], self._labels[start:end]

	def images(self):
		return self._images

	def labels(self):
		return self._labels

	def size(self):
		return self._images.shape[0]


def flatten(image):
	if (len(image.shape)) == 2:
		return image.ravel()

	data = []
	data.append(image[:,:,0].ravel())
	data.append(image[:,:,1].ravel())
	data.append(image[:,:,2].ravel())

	# data = tf.cast(np.array(data).ravel(), tf.float32)

	return np.array(data).ravel()
	# return np.array(data).ravel()

def read_dir(directory):
	data = []
	for file in os.listdir(directory):
		if file.endswith(".jpg") or file.endswith(".JPG"):
			image = cv2.imread(directory+'/'+str(file))
			data.append(image)
			data = np.array(data)
		
		print (data.shape)
	
	return data

def readDataSets(npyFiles_path="../Dataset/npyFiles/"):
	# Loading the npy array of images and labels to the array variables
	trainImages, trainLabels = np.load(npyFiles_path + "/npyTrainingImages.npy"), np.load(npyFiles_path + "/npyTrainingLabels.npy")
	testImages, testLabels = np.load(npyFiles_path + "/npyTestingImages.npy"), np.load(npyFiles_path + "/npyTestingLabels.npy")

	return Datasets(train=Dataset([trainImages, trainLabels]), test=Dataset([testImages, testLabels]))
