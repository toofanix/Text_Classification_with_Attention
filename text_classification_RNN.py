import numpy as np
import pandas as pd
from collections import defaultdict
import re
import typing
from bs4 import BeautifulSoup

import sys
import os

os.environ['KERS_BACKEND'] = 'tensorflow'

DIR_PATH = os.getcwd()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Merge, Dropout
from keras.layers import LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string: str) -> str:
	'''
	Tokenization/string cleaning for dataset
	Every dataset is lower cased except
	:param string: String to be tokenized can cleaned
	:return: tokenized/cleaned string
	'''
	string = re.sub(r"\\", "", string)
	string = re.sub(r"\'", "", string)
	string = re.sub(r"\"", "", string)
	return string.strip().lower()


# Load the data

# Load the data
data_train = pd.read_csv(
	os.path.join(DIR_PATH, 'data/labeledTrainData.tsv'),
	sep='\t')

print('Shape of data = {}'.format(data_train.shape))
print("Number of positive and negative samples = {}".format(data_train.sentiment.value_counts()))

# Collect the texts and the labels
texts = []
labels = []

for idx in range(data_train.review.shape[0]):
	text = BeautifulSoup(data_train.review[idx], "lxml")
	texts.append(clean_str(text.get_text()))
	labels.append(data_train.sentiment[idx])

# Create a tokenizer
# First fit to text to create a dictionary
# Then convert the text to sequences/numbers
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

# Pad the sequence for proper sentence length
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Convert labels to categorical
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor = {}'.format(data.shape))
print('Shape of label tensor = {}'.format((labels.shape)))

# Randomize the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_valid = data[-num_validation_samples:]
y_valid = labels[-num_validation_samples:]

print('Number of samples in train = {}'.format(x_train.shape[0]))
print('Number of samples in valid = {}'.format(x_valid.shape[0]))
print('Number of labels in train = {}'.format(y_train.shape[0]))
print('Number of labels in valid = {}'.format(y_valid.shape[0]))

print('Number of positive samples in train = {}'.format(y_train[:, 1].sum()))
print('Number of negative samples in the train = {}'.format(y_train[:, 0].sum()))

print('Number of positive samples in valid = {}'.format(y_valid[:, 1].sum()))
print('Number of negative samples in the valid = {}'.format(y_valid[:, 0].sum()))

# Load the embeddings
embeddings_index = {}
f = open(os.path.join(DIR_PATH, 'data/glove.6B.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Words in word vector = {}'.format(len(embeddings_index)))

embedding_matrix = np.random.random(
	(len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
							input_length=MAX_SEQUENCE_LENGTH, trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
preds = Dense(2, activation='softmax')(l_lstm)
model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print('Model using RNN :')
model.summary()

model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), batch_size=50)


# Attention GRU network
class AttentionLayer(Layer):
	def __init__(self, **kwargs):
		self.init = initializers.get('normal')
		super(AttentionLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3
		self.W = self.init((input_shape[-1],))
		self.trainable_weights = [self.W]
		super(AttentionLayer, self).build(input_shape)

	def call(self, x, mask=None):
		eij = K.tanh(K.dot(x, self.W))
		ai = K.exp(eij)
		weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')
		weighted_input = x * weights.dimshuffle(0, 1, 'x')
		return weighted_input.sum(axis=1)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[-1])


embedding_matrix = np.random.random(
	(len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
							input_length=MAX_SEQUENCE_LENGTH, trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttentionLayer()(l_gru)
preds = Dense(2, activation='softmax')(l_att)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print('Model with attention GRU :')
model.summary()

model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), batch_size=50)
