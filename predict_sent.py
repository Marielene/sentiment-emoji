# -*- coding: utf-8 -*-
"""


@author: Kalleid
"""

import numpy as np
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from pickle import load

def save_file(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, "w+", encoding="utf8")
	file.write(data)
	file.close()

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
in_filenameX = 'processed.txt'
docx_train = load_doc(in_filenameX)
tweets= docx_train.split('\n')

# fit text
tokenizer.fit_on_texts(tweets)
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
    final.append("100")
  elif(item[0]<item[1]>item[2]):
    final.append("010")
  else:
    final.append("001")
save_file(final, filename)