import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D, SpatialDropout1D
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
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(tweets)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(tweets)
standard=count_max(sequences)
sequences_final=sq.pad_sequences(sequences, padding='post', maxlen=standard)
unique_words=len(np.unique(sequences_final))
# organize data.
emoji_prep=prep_data_int(emoji)
X_train=sequences_final
y_train= to_categorical(emoji_prep, num_classes=None)
# Split the data

# define model
model = Sequential()
model.add(Embedding(100000, 20, input_length=standard))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(20, activation='softmax'))

# compile and print summary of model
model.compile(optimizer='ADAM', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# tuning learn rate if it stagnates through the epochs.
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.001)

# fit model and train
model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[reduce_lr], verbose=1, validation_split=0.25)

results=model.evaluate()

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
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
 
# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = model.evaluate(X_test, y_test, verbose=1)
#print("Percentage:" + ( np.mean(score)*100) + "%")