import tensorflow as tf
import numpy as np

batch_size = 128
hm_epochs = 30
rnn_size = 128
sequence_length = 80

file_path = "/Users/jamesledoux/Documents/data_exploration/author_files/shakespeare2"
text = open(file_path).read().lower()
len_text = len(text)

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
	X = np.zeros((len(sequences), maxlen, len_chars), dtype=np.float32)
	y = np.zeros((len(sequences), len_chars), dtype=np.float32)
	#X tensor is (num_sequences x sequence_length x num_characters), y matrix is (num_sequqnces x num_characters)
	for i, sequence in enumerate(sequences):
	    for t, char in enumerate(sequence):
	        X[i, t, char_to_num[char]] = 1
	    y[i, char_to_num[next_chars[i]]] = 1
	print("shape: ", X.shape)
	return X, y, char_to_num, num_to_char, len_chars


def neural_network_model(x,num_characters):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, num_characters])),
                      'biases':tf.Variable(tf.random_normal([num_characters]))}
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    val, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)
    output = tf.matmul(last,layer['weights']) + layer['biases'] #orig. had 'output' instead of 'last here'
    return output


def create_batches(X, batch_size):
	num_sequences = X.shape[0]
	num_batches = int(num_sequences / batch_size)
	batch_indexes = [j*batch_size for j in range(num_batches+1) ]
	return batch_indexes


X, y_mat, char_to_num, num_to_char, num_characters = format_data(text, sequence_length, step=3)
x = tf.placeholder('float', [None, sequence_length, num_characters])
y = tf.placeholder(tf.float32, [None, num_characters])


prediction = neural_network_model(x,num_characters)
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(hm_epochs):
		epoch_loss = 0
		batch_indexes = create_batches(X, batch_size)
		for j in range(int(X.shape[0]/batch_size)): #j == batch list index
			batch_x = X[batch_indexes[j]:batch_indexes[j+1],:]
			batch_y = y_mat[batch_indexes[j]:batch_indexes[j+1],:]
			j, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
			epoch_loss += c

		print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	print('Accuracy:',accuracy.eval({x:X, y:y_mat}))
