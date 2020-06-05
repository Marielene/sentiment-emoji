# -*- coding: utf-8 -*-
"""
Emoji-only model training

@author: Kalleid
"""

import numpy as np
from keras.preprocessing import sequence as sq
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json
from pickle import load

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

#Load tokenizer to resume training with
tokenizer = load(open('tokenizer.pkl', 'rb'))

# load data and resume training
# load prepped data
in_filenameX = 'text_train.txt'
in_filenameY = 'labels_train.txt'
docx_train = load_doc(in_filenameX)
docy_train = load_doc(in_filenameY)
tweets= docx_train.split('\n')
emoji= docy_train.split('\n')

# fit text
tokenizer.fit_on_texts(tweets)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
#move on to sequences for embedding and LSTM
sequences = tokenizer.texts_to_sequences(tweets)
#Find max length among all sequences
standard=count_max(sequences)
#pad the sequences to the max length
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

#Load model architecture from json file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
#compile and print summary
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['categorical_accuracy'])
kfold = KFold(n_splits=4)
for train, test in kfold.split(X_train, y_train):
	loaded_model.fit(X_train[train], y_train[train], epochs=1, batch_size=size_batch, verbose=1, shuffle=False, validation_data=(X_train[test],y_train[test]))

# train model further because it's real complicated and colab hates me
#Save model again after training.
# evaluate loaded model on test data
score = loaded_model.evaluate(X_test, Y_test, verbose=1)
print("Post-training:")
print("Accuracy:" + str(score[1]*100) + "%")
print("Loss:" + str(score[0]))
model_json = loaded_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("model.h5")
print("Saved model to disk")
