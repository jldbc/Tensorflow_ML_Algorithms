import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X_train = np.linspace(-1, 1, 101)
y_train = np.asmatrix(8 * X_train + np.random.randn()* 2 + 139).T
X_train = np.asmatrix(X_train).T

n_dim = X_train.shape[1]

lr = tf.constant(0.01,dtype=tf.float32)
num_epochs = 500

X = tf.placeholder(tf.float32, [X_train.shape[0],n_dim])
y = tf.placeholder(tf.float32, [X_train.shape[0],1])
w = tf.Variable(np.ones([n_dim,1]),dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
init = tf.initialize_all_variables()

yhat = tf.matmul(X,w) + b
loss = tf.reduce_mean(tf.square(yhat - y))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)


#initialize session
sess = tf.Session()
sess.run(init)

loss_history = []
for epoch in range(num_epochs):
    sess.run(train_op,feed_dict={X: X_train, y: y_train})
    loss_history.append(sess.run(loss,feed_dict={X: X_train, y: y_train}))


plt.plot(range(len(loss_history)),loss_history)
plt.show()

print("W: %.4f" % sess.run(w)) 
print("b: %.4f" % sess.run(b)) 
print("MSE: %.4f" % loss_history[-1]) 




