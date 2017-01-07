import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

batch_size = 128
hm_epochs = 10
rnn_size = 128
sequence_length = 80

"""
MNIST 
(n_observations x  num_sequences  x  sequence_length)	predicts	(n_observations x n_classes)

TEXT 
(n_sequences, sequence_length, n_characters)	predicts	(n_sequences, n_characters)
"""

def format_data(x, maxlen=80, step=3):
	"""
	read in text file, vectorize it, create translation dicts,
	break into sequences and feed into tensor
	"""
	len_text = len(x)
	chars = sorted(list(set(text)))
	len_chars = len(chars)
	print 'Text info: len {}, type {}'.format(len_text, type(x))
	#translation dicts
	char_to_num = dict((c, i) for i, c in enumerate(chars))
	num_to_char = dict((i, c) for i, c in enumerate(chars))
	sequences = []
	next_chars = []
	#split characters into sequences of a set length
	for i in range(0, len(text) - maxlen, step):
	    end_index = i + maxlen
	    sequences.append(text[i: end_index])
	    next_chars.append(text[end_index])
	print 'Total number sequences: ', len(sequences)
	# Start making your sparse matrices
	print 'Vectorizing...'
	X = np.zeros((len(sequences), maxlen, len_chars), dtype=np.float32)
	y = np.zeros((len(sequences), len_chars), dtype=np.float32)
	#X tensor is (num_sequences x sequence_length x num_characters), y matrix is (num_sequqnces x num_characters)
	for i, sequence in enumerate(sequences):
	    for t, char in enumerate(sequence):
	        X[i, t, char_to_num[char]] = 1
	    y[i, char_to_num[next_chars[i]]] = 1
	print "Done vectorizing."
	print("shape: ", X.shape)
	return X, y, char_to_num, num_to_char, len_chars

def neural_network_model(x,num_characters):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, num_characters])),
                      'biases':tf.Variable(tf.random_normal([num_characters]))}
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=False)
    print "cell construction worked"
    output, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    print "got output and state"
    output = tf.reshape(output, [-1, rnn_size]) #see if this is right
    output = tf.matmul(output,layer['weights']) + layer['biases']
    print "output generation worked"
    return output

def train_neural_network(X,y_mat,num_characters):
    prediction = neural_network_model(X,num_characters)
    print "prediction worked"
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print "cost + optimizer worked"
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape([batch_size,sequence_length,num_characters])
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape([-1, sequence_length, num_characters]), y:mnist.test.labels}))

file_path = "/Users/jamesledoux/Documents/data_exploration/author_files/shakespeare"
text = open(file_path).read().lower()
len_text = len(text)

X, y_mat, char_to_num, num_to_char, num_characters = format_data(text, sequence_length, step=3)
print "making placeholders"
x = tf.placeholder('float', [None, sequence_length, num_characters])
y = tf.placeholder('float')

train_neural_network(X,y_mat,num_characters)