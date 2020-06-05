# -*- coding: utf-8 -*-
"""
Evaluation for mixed input model

@author: Kalleid
"""
import numpy as np
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.utils import to_categorical
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
	maxi=0
	for item in count_list:
		if(maxi<len(item)):
			maxi=len(item)
	return maxi

# map integer array
def prep_data_int(raw_int):
	temp=list(map(int, raw_int))
	raw_int=np.array(temp)
	return raw_int

#Load tokenizer to resume training with
tokenizer = load(open('tokenizer.pkl', 'rb'))

# load data and resume training
# load prepped data
in_filename_tweets = 'text_test.txt'
in_filename_emoji = 'labels_test.txt'
docx_train = load_doc(in_filename_tweets)
docy_train = load_doc(in_filename_emoji)
tweets= docx_train.split('\n')
emoji= docy_train.split('\n')

# fit text
#tokenizer.fit_on_texts(tweets)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
#move on to sequences for embedding and LSTM
sequences = tokenizer.texts_to_sequences(tweets)
#pad the sequences to the max length
sequences_final=sq.pad_sequences(sequences, padding='post', maxlen=41)
unique_words=len(np.unique(sequences_final))
# organize data.
X=sequences_final
size_batch=24
emoji_prep=prep_data_int(emoji)
Y=to_categorical(emoji_prep)

#Load model architecture from json file
json_file = open('modelSent.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelSent.h5")
print("Loaded model from disk")
# compile and predict
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='RMSprop', metrics=['sparse_categorical_accuracy'])
preds=loaded_model.predict(X)
print(preds[1])
# save to preds for next step.
filename='tweets_sentiment.txt'
final=[]
for item in preds:
  if(item[0]>item[1]>item[2]):
    final.append([1,0,0])
  elif(item[0]<item[1]>item[2]):
    final.append([0,1,0])
  else:
    final.append([0,0,1])
json_file = open('model.json', 'r')
eval_model_json = json_file.read()
json_file.close()
eval_model = model_from_json(eval_model_json)
# load weights into new model
eval_model.load_weights("model.h5")
print("Loaded model from disk")
# compile and predict
eval_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['categorical_accuracy'])
X_pred=sq.pad_sequences(sequences, padding='post', maxlen=24)
X_sent=sq.pad_sequences(final, padding='post', maxlen=24)
score=eval_model.evaluate(X_pred, Y)
print('Accuracy %s' % (score[1]*100))
print('Loss: %s' % score[0])