import tensorflow as tf
import numpy as np
import sys
import cv2

from ObjectOnFloorDetectionNN.Dataset import input_data

# Reseting the graph
tf.reset_default_graph()

# Getting the data sets
data = input_data.readDataSets()
# ######################################################################################################################
# ################Lambda functions that initialize the Wights and Biases of the Neural Network##########################
set_Weights = lambda shape, stddev = 0.1: tf.Variable(tf.truncated_normal(shape, stddev = stddev))
set_Bias = lambda shape: tf.Variable(tf.constant(0.1, shape=shape))

conv2d = lambda x, W, s: tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

max_pool = lambda x, k: tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
########################################################################################################################
# Making the inputs placeholders each channels of each input picture. Each channel have 250,000 neurons or 500 pixel
# of height and 500 pixels of width for each channel (R x G x B)
x = tf.placeholder(tf.float32, shape=[50, 500, 500, 3])

# Making the output place holder of the y = x * W + b function
y_ = tf.placeholder(tf.float32, shape=[None, 250000])

################################################CONVOLUTIONAL LAYERS####################################################

# Preparing the Weights for filters of the first convolutional layer
# [filter_height, filter_width, input_image_channels, number_of_filters]
l1_W = set_Weights([3, 3, 3, 5])
l1_b = set_Bias([5]) # Setting the filters Bias
l1_output = tf.nn.relu(conv2d(x, l1_W, 1) + l1_b) # [None, 500/1, 500/1, 5] => output_size = [None, 500, 500, 5]
l1_maxpool_output = max_pool(l1_output, 2) # [None, 500/2, 500/2, 5] => output_size = [None, 250, 250, 5]

# Preparing the weights and filters of the second convolutional layer
# [filter_height, filter_width, input_past_layer_channels, number_of_filters_of_this_layer]
l2_W = set_Weights([5, 5, 5, 25])
l2_b = set_Bias([25])
l2_output = tf.nn.relu(conv2d(l1_maxpool_output, l2_W, 2) + l2_b)  # [None, 250/2, 250/2, 25] => 
                                                                # output_size = [None, 125, 125, 25]
l2_maxpool_output = max_pool(l2_output, 4) # [None, 125/4, 125/4, 25] => output_size = [None, 32, 32, 25]

# Preparing the weights and filters of the third convolutional layer
# [filter_height, filter_width, input_past_layer_channels, number_of_filters_of_this_layer]
l3_W = set_Weights([7, 7, 25, 50])
l3_b = set_Bias([50])
l3_output = tf.nn.relu(conv2d(l2_maxpool_output, l3_W, 4) + l3_b) # [None, 32/4, 32/4, 50] 
#                                                                 => output_layer = [None, 8, 8, 50]

l3_maxpool_output = max_pool(l3_output, 2) # [None, 8/2, 8/2, 50] => output = [None, 4, 4, 50]

# Flattening third convolutional layer output to enter it to the first fully connected layer
l3_output_flat = tf.reshape(l3_maxpool_output, [-1, 4 * 4 * 50])

#######################################################################################################################

#################################################FULLY CONNECTED LAYERS################################################

# Preparing the weights and biases of the first fully connected layer
l4_W = set_Weights([4 * 4 * 50, 1000000])
l4_b = set_Bias([1000000])
l4_output = tf.nn.relu(tf.matmul(l3_output_flat, l4_W) + l4_b)

# Preparing the weights and biases of the second fully connected layer
l5_W = set_Weights([1000000, 750000])
l5_b = set_Bias([750000])
l5_output = tf.nn.relu(tf.matmul(l4_output, l5_W) + l5_b)

# Preparing weights and biases of the third fully connected layer
l6_W = set_Weights([750000, 500000])
l6_b = set_Bias([500000])
l6_output = tf.nn.relu(tf.matmul(l5_output, l6_W) + l6_b)

# Preparing weights and biases of the fourth fully connected layer
l7_W = set_Weights([500000, 250000])
l7_b = set_Bias([250000])
l7_output = tf.matmul(l6_output, l7_W) + l7_b
#######################################################################################################################

# loss function
cross_entropy = -(tf.reduce_sum(y_ * tf.log(l7_output)))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy calculation
correct_prediction = tf.equal(tf.argmax(l7_output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing all the network variables

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables()) # Initializing all the network variables

  lstt = tf.trainable_variables()

  for i in range(50):
    batch = data.train.next_batch(50)

    if i % 10 == 0:
      print("Accuracy at step %i: %g" % (i, accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})))

    train_step.run(feed_dict={x:batch[0], y_:batch[1]})