import tensorflow as tf

fgs = tf.app.flags

FLAGS = fgs. FLAGS

fgs.DEFINE_float('test', 15.1, 'Initial test.')

print(FLAGS.test)