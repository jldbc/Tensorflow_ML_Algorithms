import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials import mnist
import matplotlib.pyplot as plt

num_epochs = 1000
batch_size = 300
num_hidden_nodes = 128

#input data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_train, y_train = mnist.train.next_batch(5000)
X_test, y_test = mnist.test.next_batch(100)

#define variables used in the graph
X = tf.placeholder(tf.float32, [None, 784])
w_h = tf.Variable(tf.random_normal([784,num_hidden_nodes], stddev=0.01))
w_out = tf.Variable(tf.random_normal([num_hidden_nodes,784], stddev=0.01))

#build the graph
def model(X,w_h,w_out):
	h = tf.nn.relu(tf.matmul(X,w_h))
	output = tf.matmul(h, w_out)
	return output, h

#feed forward through the model
output, h = model(X, w_h, w_out)

#loss function
cost = tf.reduce_mean(tf.square(output - X))

#optimizer function for backprop + weight updates
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(output, 1)

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(num_epochs):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		train = sess.run(train_step, feed_dict={X: batch_xs})
		if epoch % 100 == 0:
			print sess.run(cost, feed_dict = {X: batch_xs})
	#print test set results
	batch_xs, batch_ys = mnist.test.next_batch(batch_size)
	cost = sess.run(cost, feed_dict = {X: batch_xs})
	print "Final mean squared difference between test data and encoded -> decoded data: " + str(cost)
	out = sess.run(output, feed_dict = {X: batch_xs})
	for i in range(3):
		plt.imshow(batch_xs[i].reshape(28,28))
		plt.show()
		plt.imshow(out[i].reshape(28,28))
		plt.show()
	h = sess.run(h, feed_dict = {X: batch_xs}) #your hidden nodes
	#print h