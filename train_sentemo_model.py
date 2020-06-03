# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 01:38:18 2020

@author: Kalleid
"""
import numpy as np
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.models import model_from_json
from pickle import load
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

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
in_filename_sent='tweets_sentiment.txt'
in_filename_tweets = 'processed.txt'
in_filename_emoji = 'processedEmoji.txt'
docx_train = load_doc(in_filename_tweets)
docsent_train=load_doc(in_filename_sent)
docy_train = load_doc(in_filename_emoji)
tweets= docx_train.split('\n')
sentiment = docsent_train.split('\n')
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
X=sq.pad_sequences(sequences, padding='post', maxlen=standard)
unique_words=len(np.unique(X))
sent=[]
for items in sentiment:
  indices=[int(items[0])+int(items[3])]
  sent.append(indices)
X_sent=sq.pad_sequences(sent, padding='post', maxlen=standard)
dim_sent=standard
# organize data.
emoji_prep=prep_data_int(emoji)
Y=to_categorical(emoji_prep)
# split data for later validation
X_sent_test=X_sent[-1000:]
X_sent_train= X_sent[:-1000]
X_test=X[-1000:]
X_train=X[:-1000]
Y_test=Y[-1000:]
y_train=Y[:-1000]
size_batch=128

#Load model architecture from json file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# compile
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['categorical_accuracy'])

# callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
# train further
kfold = KFold(n_splits=5)
for i in range(1, 4):
    for train, test in kfold.split(X_train):
        loaded_model.fit([X_train[train], X_sent_train[train]] ,y_train[train], epochs=1, batch_size=size_batch, verbose=1, validation_data=([X_train[test],X_sent_train[test]], y_train[test]), callbacks=[reduce_lr, model_checkpoint_callback])
#Save model again after training.
model_json = loaded_model.to_json()
with open("model_sentemo.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("model_sentemo.h5")
print("Saved model to disk")
# evaluate loaded model on test data
score = loaded_model.evaluate([X_test, X_sent_test], Y_test, verbose=1)
print("Post-training:")
print("Accuracy:" + str(score[1]*100) + "%")
print("Loss:" + str(score[0]))


