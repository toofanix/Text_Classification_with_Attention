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





