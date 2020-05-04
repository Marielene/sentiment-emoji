import numpy as np
import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Activation, Dropout
from keras.layers import GlobalMaxPool1D, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json
from pickle import dump

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# find biggest list entry in a list of lists, since len only gives you the largest single element.
def count_max(count_list):
	maxboi=0
	for item in count_list:
		if(maxboi<len(item)):
			maxboi=len(item)
	return maxboi

# map integer array
def prep_data_int(raw_int):
	temp=list(map(int, raw_int))
	raw_int=np.array(temp)
	return raw_int


# load prepped data
in_filenameX = 'processed.txt'
in_filenameY = 'processedEmoji.txt'
docx_train = load_doc(in_filenameX)
docy_train = load_doc(in_filenameY)
tweets= docx_train.split('\n')
emoji= docy_train.split('\n')

# declare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(tweets)
standard=count_max(sequences)
sequences_final=sq.pad_sequences(sequences, padding='post', maxlen=standard)
unique_words=len(np.unique(sequences_final))
# organize data.
emoji_prep=prep_data_int(emoji)
X_test=sequences_final[-1000:]
X_train=sequences_final[:-1000]
Y_test=emoji_prep[-1000:]
y_train=emoji_prep[:-1000]
y_train= to_categorical(y_train, num_classes=None)
Y_test= to_categorical(Y_test, num_classes=None)
size_batch=24
# define model
#print(X_train.shape[1])
model = Sequential()
model.add(Embedding(unique_words, 32, trainable=True))
model.add(LSTM(128, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(20, activation='softmax'))

# compile and print summary of model
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# tuning learn rate if it stagnates through the epochs.
#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.001)

# fit model and train
for i in range(10):
	model.fit(X_train, y_train, epochs=1, batch_size=size_batch, verbose=1, shuffle=False)
	model.reset_states()
#model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.01, shuffle=True)
# evaluate trained model
#score = model.evaluate(X_test,Y_test)
#print("Accuracy:", score[1], "Loss:", score[0])

# save the model to file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

#eval hopefully
#load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
score = model.evaluate(X_test, Y_test, verbose=1)
print("Percentage:" + score*100 + "%")