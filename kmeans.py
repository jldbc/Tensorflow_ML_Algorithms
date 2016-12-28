import tensorflow as tf
import numpy as np

k = 4
n_obs=1000

X = np.random.uniform([n_obs,2]) #simple x,y coordinates
cluster_membership = tf.variable(tf.zeros([n_obs]), dtype=tf.int64)

