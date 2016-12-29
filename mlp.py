import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials import mnist

num_epochs = 1000
batch_size = 300

#input data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_train, y_train = mnist.train.next_batch(5000)
X_test, y_test = mnist.test.next_batch(100)

#define variables used in the graph
X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10]) 
w_h = tf.Variable(tf.random_normal([784,784], stddev=0.01))
b1 = tf.Variable(tf.zeros([784]))
w_h2 = tf.Variable(tf.random_normal([784,784], stddev=0.01))
b2 = tf.Variable(tf.zeros([784]))
w_out = tf.Variable(tf.random_normal([784,10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))

#build the graph
def model(X,y,w_h,w_h2,w_out, b1, b2, b3):
	h = tf.nn.relu(tf.matmul(X,w_h) + b1)
	h = tf.nn.dropout(h, .2)
	h2 = tf.nn.relu(tf.matmul(h,w_h2) + b2)
	h2 = tf.nn.dropout(h2, .2)
	output = tf.matmul(h2, w_out) + b3
	return output

#feed forward through the model
yhat = model(X, y, w_h, w_h2, w_out,b1, b2, b3)

#loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))

#optimizer function for backprop + weight updates
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(yhat, 1)

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(num_epochs):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		train = sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})
		if epoch % 100 == 0:
			acc = (np.mean(np.argmax(batch_ys, axis=1) == sess.run(predict_op, feed_dict={X: batch_xs, y: batch_ys})))
			print "training accuracy at epoch " + str(epoch) + ": " + str(acc)
	#print test set results
	batch_xs, batch_ys = mnist.test.next_batch(batch_size)
	acc = (np.mean(np.argmax(batch_ys, axis=1) == sess.run(predict_op, feed_dict={X: batch_xs, y: batch_ys})))
	print "Test set accuracy: " + str(acc)





