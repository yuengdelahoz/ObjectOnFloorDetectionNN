import tensorflow as tf
import numpy as np
import sys
import cv2
import functools
import operator
import time

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

# Preparing the weights and biases of the second fully connected layer
l5_W = set_Weights([25 * 25 * 13, 500 * 500])
l5_b = set_Bias([500 * 500])
l5_output = tf.nn.relu(tf.matmul(l4_output_flat, l5_W) + l5_b)

# 

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(l5_output), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy calculation
correct_prediction = tf.reduce_mean(tf.abs(tf.sub(l5_output, y_)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
print("Starting training session...")

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables()) # Initializing all the network variables

  lstt = tf.trainable_variables()
    
  [print (lt.get_shape()) for lt in lstt]

  acum = 0

  for lt in lstt:
    ta = lt.get_shape()
    lstd = ta.as_list()
    mult = functools.reduce(operator.mul, lstd, 1)
    acum = acum + mult
    
  print("Number of parameters: ", acum)
  
  for i in range(1000):

    if i == 0:
      lst = []

      W2 = np.zeros([28, 28])
      W = l5_output.eval(sess)

      for i in range(5):
        for j in range(28):
          for k in range(28):
            W2[j][k] = W[j][k][0][i]
        max = np.max(W2)
        min = np.min(W2)

        W2 = (W2-min)/(max-min)
        W2 = W2 * 255
        
        lst.append(W2.copy())
      [cv2.imwrite('output_' + str(i) + '_before.png', lst[i]) for i in range(len(lst))]

    print("Iteration " + str(i) + " took: ", end="")
    start = time.time()
    
    batch = data.train.next_batch(50)

    train_step.run(feed_dict={x:batch[0], y_:batch[1]})
    
    end = (time.time() - start) /60

    print(str(end) + " segs")

    if i % 100 == 0:
      print("Accuracy at step %i: %g" % ((i), accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})))
      print(str(end) + " segs")

      save_path = saver.save(sess, "/home/a1mb0t/Documents/FloorDetectionNN.ckpt", global_step=i)