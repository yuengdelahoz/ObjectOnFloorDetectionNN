import tensorflow as tf
import numpy as np
import sys
import cv2
import functools
import operator
import time
import os

from ObjectOnFloorDetectionNN.input_data import input_data

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
l1_W = set_Weights([28, 28, 3, 5])
l1_b = set_Bias([5]) # Setting the filters Bias
l1_output = tf.nn.relu(conv2d(x, l1_W, 1) + l1_b) # [None, 500/1, 500/1, 5] => output_size = [None, 500, 500, 5]

# Preparing the weights and filters of the second convolutional layer
# [filter_height, filter_width, input_past_layer_channels, number_of_filters_of_this_layer]
l2_W = set_Weights([21, 21, 5, 9])
l2_b = set_Bias([9])
l2_output = tf.nn.relu(conv2d(l1_output, l2_W, 2) + l2_b)  # [None, 500/2, 500/2, 9] => 
                                                                # output_size = [None, 250, 250, 9]

# Preparing the weights and filters of the third convolutional layer
# [filter_height, filter_width, input_past_layer_channels, number_of_filters_of_this_layer]
l3_W = set_Weights([14, 14, 9, 11])
l3_b = set_Bias([11])
l3_output = tf.nn.relu(conv2d(l2_output, l3_W, 5) + l3_b) # [None, 250/5, 250/5, 11] 
#                                                                 => output_layer = [None, 50, 50, 11]

# Preparing fourth convolutional layer
l4_W = set_Weights([7, 7, 11, 13])
l4_b = set_Bias([13])
l4_output = tf.nn.relu(conv2d(l3_output, l4_W, 2) + l4_b)  # [None, 50/2, 50/2, 13] => [None,  25, 25, 13]

l4_output_flat = tf.reshape(l4_output, [-1, 25 * 25 * 13])
#######################################################################################################################

#################################################FULLY CONNECTED LAYERS################################################
keep_prob = tf.placeholder(tf.float32)
l4_output_drop = tf.nn.dropout(l4_output_flat, keep_prob)

# Preparing the weights and biases of the second fully connected layer
l5_W = set_Weights([25 * 25 * 13, 500 * 500])
l5_b = set_Bias([500 * 500])
l5_output = tf.nn.relu(tf.matmul(l4_output_drop, l5_W) + l5_b)
#######################################################################################################################

# loss function
cross_entropy = tf.reduce_reduce(-mean.tf_sum(y_ * tf.log(l5_output), reduction_indices=[1]))

# Optimizer of the network
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy calculation
accuracy = tf.reduce_mean(tf.cast(tf.reduce_mean(tf.abs(tf.sub(l5_output, y_))), tf.float32))

# Variable utilized to save the network values into a file and restore the values of that file for the training session
saver = tf.train.Saver()

# List of files of the network variables for the past training sessions
outputFiles = sorted(os.listdir("ObjectOnFloorDetectionNN/NetworkValues/"))

print("Starting training session...")

with tf.Session() as sess:
  # Calculating the new step for saving the new network files
  step = len(outputFiles) - 1

  if len(outputFiles) == 0: # verifying if there are old files of the network variables
    # Restoring the values of the variables of the last training session
    saver.restore(sess, "ObjectOnFloorDetectionNN/NetworkValues/" + str(outputFiles[-1]))
  else:
    sess.run(tf.initialize_all_variables()) # Initializing all network variables if no network files
  
  # Variable for calculate the total number of network parameters
  acum = 0

  print("Calculating the number of network's parameters...")
  for lt in tf.trainable_variables():
    mult = functools.reduce(operator.mul, lt.get_shape().as_list(), 1)
    acum = acum + mult
    
  print("\n\tNumber of parameters: %g" % acum)

  print("\n\nStarting the training loop...")
  for i in range(1000):
    print("\nCurrent iteration " + str(i) + "\n")
    batch = data.train.next_batch(50)

    if i == 0:
      print("\n\tEvaluating the output of the network...\n")
      t = l5_output.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1})

      print("Resizing the images...")
      img_Ouput = t[0].reshape((500,500))
      img_Input = batch[0][0]
      img_Label = batch[1][0].reshape((500,500))

      print("\n\tSaving the output of images...\n")
      cv2.imwrite('img_Ouput.jpeg', img_Ouput)
      cv2.imwrite('img_Input.jpeg', img_Input)
      cv2.imwrite('img_Label.jpeg', img_Label)

    elif i == 999:
      print("\n\tEvaluating the output of the network...\n")
      t = l5_output.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1})

      print("Resizing the images...")
      img_Ouput = t[0].reshape((500,500))
      img_Input = batch[0][0]
      img_Label = batch[1][0].reshape((500,500))

      print("\n\tSaving the output of images...\n")
      cv2.imwrite('img_Ouput.jpeg', img_Ouput)
      cv2.imwrite('img_Input.jpeg', img_Input)
      cv2.imwrite('img_Label.jpeg', img_Label)

    start = time.time()

    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    print("\nIteration " + str(i) + " took: %.2f mins" % ((time.time() - start) / 60))

    if i % 100 == 0:
      print("Accuracy at step %i: %.2f%" % ((i), accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})))
      
      save_path = saver.save(sess, "ObjectOnFloorDetectionNN/NetworkValues/networkValues.ckpt", global_step=(step + i))

      print("File saved to " + str(save_path))