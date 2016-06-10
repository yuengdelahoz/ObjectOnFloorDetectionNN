import os, os.path
import sys
import numpy as np
import tensorflow as tf
import cv2
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

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

def check_dir(dir1, dir2):
	# This methods checks if the content of both dir is of the same size
	dir1_len = len([name for name in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, name))])
	dir2_len = len([name for name in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, name))])
	# if one of the two directories is empty the program stops
	if dir1_len == dir2_len and (dir1_len!=0 and dir2_len!=0):
		return True
	raise ValueError("Dataset error...number of files in training or testing directories are not the same")

def read_dir(directory):
	data = []
	for file in os.listdir(directory):
		if file.endswith(".jpg") or file.endswith(".JPG"):
			image = cv2.imread(directory+'/'+str(file))
			data.append(image)
	data = np.array(data)
	print (data.shape)
	return data

def check_and_read(dir1, dir2):
	if check_dir(dir1, dir2):
		return read_dir(dir1), read_dir(dir2)

def readDataSets():
	# get path of current file (input_data.py)
	directory = os.path.dirname(__file__)
	# read traing images and labels from folder "datasets"
	data1 = check_and_read(directory + '/datasets/training_data/images', directory + '/datasets/training_data/labels')
	# read testing images and labels from folder "datasets"
	data2 = check_and_read(directory + '/datasets/test_data/images', directory + '/datasets/test_data/labels')
	#Creating two datasets objects. One for traingin and another for testing.
	train = Dataset(data1)
	testing = Dataset(data2)
	return Datasets(train=train,test=testing)


class Dataset:
	def __init__(self,images):
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
			p.random.shuffle(temp)
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
