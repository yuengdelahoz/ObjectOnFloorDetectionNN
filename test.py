import tensorflow as tf
import cv2

set_Weights = lambda shape, stddev = 0.1: tf.Variable(tf.truncated_normal(shape, stddev = stddev))
set_Bias = lambda shape: tf.Variable(tf.constant(0.1, shape=shape))

l4_output = set_Weights([25, 25, 13])

l4_output_flat = tf.reshape(l4_output, [-1, 25 * 25 * 13])

l5_W = set_Weights([25 * 25 * 13, 500 * 500])
l5_b = set_Bias([500 * 500])
l5_output = tf.nn.relu(tf.matmul(l4_output_flat, l5_W) + l5_b)


# print("Shape of l4_output " + str(l4_output.get_shape()))
# print("Shape of l4_output_flat " + str(l4_output_flat.get_shape()))
# print("Shape of l5_W " + str(l5_W.get_shape()))
# print("Shape of l5_b " + str(l5_b.get_shape()))
# print("Shape of l5_output " + str(l5_output.get_shape()))

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()
# Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  print("Session initiated...")
  sess.run(init_op)

  t = l5_output.eval(sess)

  print(t.get_shape())

  # l5_output_npArray = tf.contrib.util.make_ndarray(tf.reshape(, [500, 500]))

  # print("Shape of l5_output_npArray " + str(l5_output_npArray.shape))

  # cv2.imwrite('test.jpeg', l5_output_npArray)