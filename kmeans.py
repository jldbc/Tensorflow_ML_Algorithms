import tensorflow as tf
import numpy as np

k = 4
n_obs=1000
num_iter = 1000

X_mat = np.random.random([n_obs,2])

X = tf.placeholder(tf.float32, [X_mat.shape[0], X_mat.shape[1]])

"""
Init:
X: [n x 2]
centroids: [k x 2]

reshape:
X: [n x k x 2]    * same as the n x 2 representation, but repeat each observation k times in the k x 2
centroids: [n x k x 2]   * each kx2 is the centroid coordinates. repeat these n times.

get distances to centroids:
reduce_sum of squared dist between reshaped X and centroids. Each k x 2 reduced to k distances
assignments = Arg_min(distances) => [n x 1], where each observation is the ID of the centroid with lowest distance 

update centroids:
coords = mean(data where k col == k) for each k  // (tf.unsorted_segment_sum(data, centroids, k) ) / tf.ones(n,k,2)
"""

cluster_membership = tf.Variable(tf.zeros([n_obs]), dtype=tf.float32)

centroids = tf.Variable(tf.random_uniform([k,2]), dtype=tf.float32)

X_temp = tf.reshape(tf.tile(X, [1,k]), [n_obs, k, 2])
centroids_temp = tf.reshape(tf.tile(centroids,[n_obs,1]), [n_obs, k, 2])

distances_to_centroids = tf.reduce_sum(tf.square(tf.sub(X_temp, centroids_temp)), reduction_indices=2)  #N x k x 1

cluster_membership = tf.arg_min(distances_to_centroids, 1) #distance-minimizing column for each row

new_means_numerator = tf.unsorted_segment_sum(X, cluster_membership, k)
new_means_denominator = tf.unsorted_segment_sum(tf.ones_like(X), cluster_membership, k)
new_means = new_means_numerator/new_means_denominator

update_centroids = centroids.assign(new_means)


init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for i in range(num_iter):
		centroids = sess.run(update_centroids, feed_dict={X:X_mat})
		print centroids
