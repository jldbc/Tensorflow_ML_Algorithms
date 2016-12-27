import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#let's use MNIST for this
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X_train, y_train = mnist.train.next_batch(5000)
X_test, y_test = mnist.test.next_batch(100)

target_x = tf.placeholder("float", [1,784]) #target vector
X = tf.placeholder("float", [None, 784]) #matrix of observations to compare to target
y = tf.placeholder("float", [None, 10]) #matrix of one-hot class vectors 

l1_dist= tf.reduce_sum(tf.abs(tf.sub(X, target_x)), 1)  #euclidean distance. the sum of squared differences between elements, row-wise.

l2_dist = tf.reduce_sum(tf.square(tf.sub(X, target_x)), 1)  #euclidean distance. the sum of squared differences between elements, row-wise.

"""
next: make this a loop to allow k>1.
arg min distance. append class to list. slice that observation 
from X and Y. Repeat K times to get the KNN. 
""" 
nn = tf.argmin(l1_dist, 0)

init = tf.initialize_all_variables()
 
accuracy_history = []
with tf.Session() as sess:
	sess.run(init)
	for obs in range(X_test.shape[0]):
		nn_index = sess.run(nn, feed_dict = {X: X_train, y: y_train, target_x: np.asmatrix(X_test[obs])})
		nn_class = np.argmax(y_train[nn_index])
		true_class = np.argmax(y_test[obs])
		print "True class: " + str(true_class) + ", predicted class: " + str(nn_class)
		if nn_class == true_class:
			accuracy_history.append(1)
		else:
			accuracy_history.append(0)

print "model was " + str(np.mean(accuracy_history)) + "% accurate"