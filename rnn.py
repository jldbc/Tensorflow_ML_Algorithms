import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
batch_size = 128
hm_epochs = 10
chunk_size = 28
num_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, num_chunks, chunk_size])
y = tf.placeholder('float')

def neural_network_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, num_chunks, x)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    output, state = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(output[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape([batch_size,num_chunks,chunk_size])
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape([-1, num_chunks, chunk_size]), y:mnist.test.labels}))

train_neural_network(x)