import tensorflow as tf
import numpy as np
import sys
from WaterDetection.Dataset import input_data

data = input_data.readDataSets()
trainImgs = data.train.images()
trainLbls = data.train.labels()
testImgs = data.test.images()
testLbls = data.test.labels()
# x = tf.placeholder(tf.float32, shape=[-1,1500,1500,3])
