import tensorflow as tf
import pandas as pd
import numpy as np

#threshold for stopping (at what value of sum sq. diffs will we consider this converged?)
threshold = 0.001
#prob that the random surfer teleports 
#see: https://www.math.upenn.edu/~kazdan/312F12/JJ/MarkovChains/markov_google.pdf
surfer_prob = 0.15
#read in and format data
#path = "/path/to/data"
#X = pd.read_csv(path)
X = np.array([[0,0,.3333,0,.333,0,.25],
 			 [0,.5,.3333,0,0,.5,0],
			 [1,0,.3333,.3333,0,0,0],
			 [0,0,0,.3333,.333,0,0],
			 [0,0,0,0,0,0,.25],
			 [0,0,0,0,0,.5,.25],
			 [0,.5,0,.3333,.333,0,.25]])
nrow = X.shape[0]
ncol = X.shape[1]
X2 = np.multiply(np.ones([nrow, ncol]), (surfer_prob/nrow))
X = np.add(np.multiply((1-surfer_prob),X), X2)
val = 1./nrow
vect = np.ones(nrow).reshape(nrow,1)

if nrow != ncol:
	print "Error: must pass the algorithm a square matrix"

A = tf.mul(tf.placeholder(tf.float32, [nrow, ncol]), (1-surfer_prob))
v = tf.placeholder(tf.float32, [ncol, 1])
prev_vector = v
pagerank_vector = tf.matmul(A,v)
change_from_iteration = tf.reduce_sum(tf.square(tf.sub(pagerank_vector, v)), reduction_indices=0) #verify that this is the correct reduction index

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	change = float('inf')#sess.run(change_from_iteration, feed_dict={A:X, v:vect})
	while(change > threshold):
		new_vect = sess.run(pagerank_vector, feed_dict={A:X, v:vect}) #update V
		change = sess.run(change_from_iteration, feed_dict={pagerank_vector:new_vect, v:vect})
		vect = new_vect
		#print v.get_shape()
		print vect
		print change
	pr_vector = sess.run(pagerank_vector, feed_dict={A:X, v:vect})
	print pr_vector
