import numpy as np
import pandas as pd
from collections import defaultdict
import re
import typing
from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

DIR_PATH = os.getcwd()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Merge, Dropout
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
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


# Loa the data
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
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
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

convs = []
filter_sizes = [3, 4, 5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
	l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedding_sequences)
	l_pool = MaxPooling1D(5)(l_conv)
	convs.append(l_conv)

l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_conv1 = Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_conv1)
l_conv2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_conv2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print('Model using a complex convolutional neural network :')
model.summary()

model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), batch_size=50)