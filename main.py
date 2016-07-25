import tensorflow as tf
import numpy as np
import sys
import cv2
import functools
import operator
import time
import os

from ObjectOnFloorDetectionNN.input_data import input_data

networkSavingFilesPath = "/home/harry/Documents/Gits/YuengGit/ObjectOnFloorDetectionNN/Dataset/NetworkValues"
indexList = list()
# Reseting the graph
tf.reset_default_graph()

# Getting the data sets
data = input_data.readDataSets()
# ######################################################################################################################
# ################Lambda functions that initialize the Wights and Biases of the Neural Network##########################
set_Weights = lambda shape, name, stddev = 0.1: tf.Variable(tf.truncated_normal(shape, stddev = stddev), name=name)
set_Bias = lambda shape, name: tf.Variable(tf.constant(0.1, shape=shape), name=name)

conv2d = lambda x, W, s: tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

max_pool = lambda x, k: tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
########################################################################################################################
# Making the inputs placeholders each channels of each input picture. Each channel have 250,000 neurons or 500 pixel
# of height and 500 pixels of width for each channel (R x G x B)
x = tf.placeholder(tf.float32, shape=[50, 500, 500, 3])

# Making the output place holder of the y = x * W + b function
y_ = tf.placeholder(tf.float32, shape=[None, 1000])

################################################CONVOLUTIONAL LAYERS####################################################

# Preparing the Weights for filters of the first convolutional layer
# [filter_height, filter_width, input_image_channels, number_of_filters]
l1_W = set_Weights([28, 28, 3, 5], "l1_W")
l1_b = set_Bias([5], "l1_b") # Setting the filters Bias
l1_output = tf.nn.relu(conv2d(x, l1_W, 1) + l1_b) # [None, 500/1, 500/1, 5] => output_size = [None, 500, 500, 5]

# Preparing the weights and filters of the second convolutional layer
# [filter_height, filter_width, input_past_layer_channels, number_of_filters_of_this_layer]
l2_W = set_Weights([21, 21, 5, 9], "l2_W")
l2_b = set_Bias([9], "l2_b")
l2_output = tf.nn.relu(conv2d(l1_output, l2_W, 2) + l2_b)  # [None, 500/2, 500/2, 9] => 
                                                                # output_size = [None, 250, 250, 9]

# Preparing the weights and filters of the third convolutional layer
# [filter_height, filter_width, input_past_layer_channels, number_of_filters_of_this_layer]
l3_W = set_Weights([14, 14, 9, 11], "l3_W")
l3_b = set_Bias([11], "l3_b")
l3_output = tf.nn.relu(conv2d(l2_output, l3_W, 5) + l3_b) # [None, 250/5, 250/5, 11] 
#                                                                 => output_layer = [None, 50, 50, 11]

# Preparing fourth convolutional layer
l4_W = set_Weights([7, 7, 11, 13], "l4_W")
l4_b = set_Bias([13],  "l4_b")
l4_output = tf.nn.relu(conv2d(l3_output, l4_W, 5) + l4_b)  # [None, 50/5, 50/5, 13] => [None,  10, 10, 13]

l4_output_flat = tf.reshape(l4_output, [-1, 10 * 10 * 13])
#######################################################################################################################

#################################################FULLY CONNECTED LAYERS################################################
keep_prob = tf.placeholder(tf.float32)
l4_output_drop = tf.nn.dropout(l4_output_flat, keep_prob)

# Preparing the weights and biases of the second fully connected layer
l5_W = set_Weights([10 * 10 * 13, 1000], "l5_W")
l5_b = set_Bias([1000], "l5_b")
l5_output = tf.nn.relu(tf.matmul(l4_output_drop, l5_W) + l5_b)
#######################################################################################################################

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(l5_output), reduction_indices=[1]))

# Optimizer of the network
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy calculation
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))

# Variable utilized to save the network values into a file and restore the values of that file for the training session
saver = tf.train.Saver()

# List of files of the network variables for the past training sessions
outputFiles = sorted(os.listdir(networkSavingFilesPath))

print("Starting training session...")

with tf.Session() as sess:

  sess.run(tf.initialize_all_variables())

  # Calculating the new step for saving the new network files
  step = len(outputFiles) - 1

  if len(outputFiles) > 0: # verifying if there are old files of the network variables
    # Restoring the values of the variables of the last training session
    saver.restore(sess, networkSavingFilesPath + str(outputFiles[-1]))
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
  for i in range(500):
    print("\nCurrent iteration " + str(i) + "\n")
    batch = data.train.next_batch(50)

    start = time.time()

    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    print("\nTraining Step " + str(i) + " took: %.2f mins" % ((time.time() - start) / 60))

    save_path = saver.save(sess, networkSavingFilesPath + "/NetworkValuesnetworkValues.ckpt", global_step=(step + i))

    print("File saved to " + str(save_path))
    if i % 100 == 0:
      t = l5_output.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1})
      save_Output(t, i)
      save_SuperImage(batch[1])
      blend_Saver(x, i, t)

      print("Accuracy at training step %i: %.2f%%" % ((i), accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})))
      
      save_path = saver.save(sess, "ObjectOnFloorDetectionNN/Dataset/NetworkValues/NetworkValuesnetworkValues.ckpt", global_step=(step + i))

      print("File saved to " + str(save_path))

  print("\n\n\tStarting testing of the net...\n\n")
  batch = data.test.next_batch(50)

  print("Accuracy with test set: %.2f%%" % (accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})))

get_Indexes = lambda output: [indexList.append(i) for i in range(len(output)) if l[i] == 1]

save_Output = lambda output, i: np.save(networkSavingFilesPath + "/output_iter_" + str(i), np.asarray(output))

save_SuperImage = lambda SuperImage, i: np.save(networkSavingFilesPath + "/superImage_iter_" + str(i), np.asarray(SuperImage))

def blend_Saver(inputBatch, _iter, output):
  print("\n\tGetting indexes...")
  get_Indexes(output)
  print("\n\tIndexes done!")
  for (img, k) in zip(inputBatch, range(len(inputBatch))):
    print("\n\tSaving input image...")
    cv2.imwrite(networkSavingFilesPath + "input_iter_" + str(_iter) + "_" + str(k) + ".jpeg", img)
    print("\n\tInput image saved!!")
    for x in range(img.shape[0]):
      for y in range(img.shape[1]):
        index = int(y // (img.shape[0] // 10)) + (x // (img.shape[1] // 100))

        if index in indexList:
          img[x, y, 0] = img[x, y, 0] * 0.7 + 255 * 0.3
    print("\n\tSaving out image...")
    cv2.imwrite(networkSavingFilesPath + "/result_iter_" + str(_iter) + "_" + str(k) +".jpeg", img)
    print("\n\tInput out saved!!")

    indexList = list()