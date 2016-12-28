import tensorflow as tf
import numpy as np
from scipy import stats
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X_train, y_train = mnist.train.next_batch(5000)
X_test, y_test = mnist.test.next_batch(100)

k = 3

target_x = tf.placeholder("float", [1,784]) #target vector
X = tf.placeholder("float", [None, 784]) #matrix of observations to compare to target
y = tf.placeholder("float", [None, 10]) #matrix of one-hot class vectors 

l1_dist= tf.reduce_sum(tf.abs(tf.sub(X, target_x)), 1)  #euclidean distance. the sum of squared differences between elements, row-wise.

l2_dist = tf.reduce_sum(tf.square(tf.sub(X, target_x)), 1)  #euclidean distance. the sum of squared differences between elements, row-wise.

#nn = tf.argmin(l1_dist, 0)
nn = tf.nn.top_k(-l1_dist, k)

init = tf.initialize_all_variables()
accuracy_history = []
with tf.Session() as sess:
	sess.run(init)
	for obs in range(X_test.shape[0]):
		nn_index = sess.run(nn, feed_dict = {X: X_train, y: y_train, target_x: np.asmatrix(X_test[obs])})
		pred_classes = []
		for i in range(k):
			nn_class = np.argmax(y_train[nn_index[1][i]])
			#print nn_class
			pred_classes.append(nn_class)
		predicted_class = stats.mode(pred_classes)[0][0]
		true_class = np.argmax(y_test[obs])
		print "True class: " + str(true_class) + ", predicted class: " + str(predicted_class)
		if predicted_class == true_class:
			accuracy_history.append(1)
		else:
			accuracy_history.append(0)

print "model was " + str(np.mean(accuracy_history)) + "% accurate"