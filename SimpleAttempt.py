
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
from keras.layers import GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.backend import shape
# fix random seed for reproducibility
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
data={}
# load prepped data
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
#tokenizer.fit_on_sequences(tweets)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(tweets)
unique_words=len(np.unique(sequences))
print("unique words: " + str(unique_words))
# standardize input
standard=0
for sequence in sequences:
	if(standard<len(sequence)):
		standard=len(sequence)
X_train=sequences
X_train = sq.pad_sequences(X_train, padding='post', maxlen=standard)
y_train=emoji
print(y_train.shape)
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=standard))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
#print(y_train[0] + " " + X_train[0])
model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
