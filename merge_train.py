# -*- coding: utf-8 -*-
"""
Two input two output model, with combined lower half.

@author: Kalleid
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random # for shuffle
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence as sq
import re
from keras.utils import to_categorical
from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
from sklearn.model_selection import KFold
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
lemmatizer = WordNetLemmatizer();

 # Load data sent
cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
data = data[['text','sentiment']]
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z\s]','',x)))
data['text'] = data['text'].apply(lambda x: x.lower())
data['text']= data['text'].apply((lambda x: lemmatizer.lemmatize(x)))
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')

#Load tokenizer to resume training with
tokenizer = load(open('tokenizer.pkl', 'rb'))

# load data and resume training
# load prepped data
in_filenameX = 'processed.txt'
in_filenameY = 'processedEmoji.txt'
docx_train = load_doc(in_filenameX)
docy_train = load_doc(in_filenameY)
tweets= docx_train.split('\n')
emoji= docy_train.split('\n')

# fit text
the_sacred_texts= list(data['text'].values)
for tweet in tweets:
	the_sacred_texts.append(tweet)
# declare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(the_sacred_texts)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
sequences_emo = tokenizer.texts_to_sequences(tweets)
sequences_sent = tokenizer.texts_to_sequences(data['text'].values)
standard=50
X_emoji=sq.pad_sequences(sequences_emo, padding='post', maxlen=standard)
X_sent= sq.pad_sequences(sequences_sent, padding='post', maxlen=standard)
word_index=tokenizer.word_index
# organize data.
emoji_prep=prep_data_int(emoji)
Y_emoji=to_categorical(emoji_prep)
Y_sent = np.array(data['sentiment'].values)
Y_sent = np.where(Y_sent==4, 2, Y_sent)
Y_sent = np.where(Y_sent==2, 1, Y_sent)
#Bruh
indices = np.arange(X_sent.shape[0])
random.shuffle(indices)
X_sent = X_sent[indices]
Y_sent = Y_sent[indices]
X_sent= X_sent[650000:1150000]
Y_sent= Y_sent[650000:1150000]
# mini-eval batch
X2=X_sent[-1000:]
X_sent= X_sent[:-1000]
X1=X_emoji[-1000:]
X_emoji=X_emoji[:-1000]
Y1=Y_emoji[-1000:]
Y_emoji=Y_emoji[:-1000]
Y2=Y_sent[-1000:]
Y_sent=Y_sent[:-1000]
size_batch=128

#Load model architecture from json file
json_file = open('model_Merged.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_Merged_weights.h5")
print("Loaded model from disk")
#compile and add summary
loaded_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.05),
    loss={'sent_output': keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="auto", name="sparse_categorical_crossentropy"),
          'emo_output': keras.losses.CategoricalCrossentropy(from_logits=True, reduction="auto", name="sparse_categorical_crossentropy")},
    metrics={'sent_output': keras.metrics.SparseCategoricalAccuracy(),
            'emo_output': keras.metrics.CategoricalAccuracy()})
print(loaded_model.summary())
# crossvalidation
kfold= Kfold(n_splits=5)
for train, test in kfold.split(X_sent):
    loaded_model.fit([X_emoji,X_sent], [Y_emoji, Y_sent], shuffle=True,
           batch_size=size_batch, epochs=8, validation_split=0.1)
#Save model again after training.
model_json = loaded_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("model_Merged_weights.h5")
print("Saved model to disk")
# evaluate loaded model on test data
score = loaded_model.evaluate([X1,X2], [Y1,Y2], verbose=1)
print("Post-training:")
print("Accuracy:" + str(score[1]*100) + "%")
print("Loss:" + str(score[0]))