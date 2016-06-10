from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import cv2
import time
import operator
import functools

def timing(f,*args):
	time1 = time.time()
	f(*args)
	time2 = time.time()
	print('{0} function took {1:0.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

 #Functions to initialize weights and biases
def W_init(shape):
	# generate random numbers from a truncated normal distribution
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def B_init(shape):
	# initilize bias as a constant.
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W,s):
	return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

def max_pool_kxk(x,k):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1],strides=[1, k, k, 1], padding='SAME')

class Network:
	def __init__(self, dataset):
		self.dataset = dataset

	def run (reps=100):
		#Reseting Graph
		tf.reset_default_graph()
		# placeholder for input and prediction
		x = tf.placeholder(tf.float32, shape=[1,500,500,3])
		y_ = tf.placeholder(tf.float32, shape=[None, 10])

		# Weights and biase for 1st conv layer
		# shape first layer [filter_height,filter_width,image channels, number of feature maps]
		l1_shape = [3,3,3,5] # 5 feature maps
		l1_W = W_init(l1_shape)
		l1_B = B_init([5])
		l1_output = tf.nn.relu(conv2d(x,l1_W,1) + l1_B)# output size = [?,28/1,28/1,10]=[?,28,28,10]
		l1_out_maxpool = max_pool_kxk(l1_output,2)# output size = [?,28/2,28/2,10] = [?,14,14,10]
		# flattening feature maps

		# Weights and bias for 2nd conv layer
		l2_shape = [5,5,5,32] # 10 features maps from previous layer, 32 new feature maps
		l2_W = W_init(l2_shape)
		l2_B = B_init([32])
		l2_output = tf.nn.relu(conv2d(l1_out_maxpool,l2_W,2) + l2_B)# output size = [?,14/2,14/2,32] = [?,7,7,32]
		l2_out_maxpool = max_pool_kxk(l2_output,4)# output size = [?,7/4,7/4,32] = [?,2,2,32]

		# flattening feature maps
		l2_out_act_max_flat = tf.reshape(l2_out_maxpool, [-1,2*2*32])

		# Weights and bias for the 1rst fully connected layer
		l3_shape = [2*2*32,100]
		l3_W = W_init(l3_shape)
		l3_B = B_init([100])
		l3_output = tf.nn.relu(tf.matmul(l2_out_act_max_flat,l3_W) + l3_B)

		# Weights and bias for the 2nd fully connected layer
		l4_shape = [100,10]
		l4_W = W_init(l4_shape)
		l4_B = B_init([10])
		l4_output = tf.matmul(l3_output,l4_W) + l4_B
		y_conv = tf.nn.softmax(l4_output)

		# loss function
		cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		# accuracy calculation
		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		init = tf.initialize_all_variables()
		# Creating session and initilizing variables
		with tf.Session() as sess2:
			sess2.run(init)
			lstt = tf.trainable_variables()
			loss = []
			for i in range(reps):
				batch = mnist.train.next_batch(50)
				if i%100 == 0:
					train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
					loss.append(train_accuracy)
					print("step %d, training accuracy %g"%(i, train_accuracy))
				train_step.run(feed_dict={x:batch[0],y_:batch[1]})
			print('Accuracy in testing set',accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
