import re
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.backend import shape
from keras import optimizers
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
#find biggest LIST entry
def countMax(count_list):
	maxboi=0
	for item in count_list:
		if(maxboi<len(item)):
			maxboi=len(item)
	return maxboi
# load prepped data
#in_filenameX = 'processedSmallBatch.txt'
#in_filenameY = 'processedEmojiSmallBatch.txt'
in_filenameX = 'processed.txt'
in_filenameY = 'processedEmoji.txt'
docTweets = load_doc(in_filenameX)
docEmojis = load_doc(in_filenameY)
emoji= docEmojis.split('\n')
tweets= docTweets.split('\n')
temp=list(map(int, emoji))
emoji=np.array(temp)
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(tweets)
unique_words=len(np.unique(sequences))
print("unique words: " + str(unique_words))
# standardize input
standard=countMax(sequences)
X_train=sequences
X_train = sq.pad_sequences(X_train, padding='post', maxlen=standard)
y_train=emoji
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=standard))
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer=sgd, loss='kullback_leibler_divergence', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=128, epochs=32, verbose=1, validation_split=0.2)
# Final evaluation of the model

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))