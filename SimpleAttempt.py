import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import MaxPooling1D, Conv1D
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
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
#unique_words=len(np.unique(sequences))
standard=count_max(sequences)
sequences_final=sq.pad_sequences(sequences, padding='post', maxlen=standard)
# organize data.
delimeter_train=32000000
X_train = sequences_final[:delimeter_train, :]
X_test=sequences_final[delimeter_train:, :]
emoji_prep=prep_data_int(emoji)
y= to_categorical(emoji_prep, num_classes=None)
y_train, y_test = y[:delimeter_train], y[delimeter_train:]
# define model
model = Sequential()
# embed list of words
model.add(Embedding(vocab_size, 100, input_length=standard))
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
model.fit(X_train, y_train, epochs=32, batch_size=1024, callbacks=[reduce_lr], verbose=1)
# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
# Final evaluation of the model
print("Evaluation section, please stand by.")
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X_test, y_test, cv=kfold)
print("Baseline:" + str(results.mean()*100) +" ("+ str(results.std()*100) +")")