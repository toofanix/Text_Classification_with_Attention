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

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Merge, Dropout
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from nltk import tokenize

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
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
data_train = pd.read_csv(
	os.path.join(DIR_PATH, 'data/labeledTrainData.tsv'),
	sep='\t')

print('Shape of data = {}'.format(data_train.shape))
print("Number of positive and negative samples = {}".format(data_train.sentiment.value_counts()))

reviews = []
labels = []
texts = []

for idx in range(data_train.review.shape[0]):
	text = BeautifulSoup(data_train.review[idx], "lxml")
	text = clean_str(text.get_text())
	texts.append(text)
	sentences = tokenize.sent_tokenize(text)
	reviews.append(sentences)
	labels.append(data_train.sentiment[idx])

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)

data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
	for j, sent in enumerate(sentences):
		if j < MAX_SENTS:
			word_tokens = text_to_word_sequence(sent)
			k = 0
			for _, word in enumerate(word_tokens):
				if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NUM_WORDS:
					data[i, j, k] = tokenizer.word_index[word]
					k = k + 1


word_index = tokenizer.word_index
print('Total unique tokens = {}'.format(len(word_index)))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor = {}'.format(data.shape))
print('Shape of label tesnor = {}'.format(labels.shape))

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
							input_length=MAX_SENT_LENGTH, trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
sent_encoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sent_encoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
preds = Dense(2, activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)


