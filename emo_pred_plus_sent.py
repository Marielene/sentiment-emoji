# -*- coding: utf-8 -*-
"""
Combined model using sentiment vector and tweet data to predict emoji

@author: Kalleid
"""
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input, Bidirectional, Dropout, concatenate, GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras.models import model_from_json
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
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

# setup vars
EMBEDDING_DIM = 50
dim_sent=0
# crossvalidator
kfold = KFold(n_splits=5)

#load embedding data
print('Indexing word vectors.')

embeddings_index = {}
with open('glove.twitter.27B.50d.txt', encoding='latin-1') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# load prepped data
in_filename_sent='tweets_sentiment.txt'
in_filename_tweets = 'text_train.txt'
in_filename_emoji = 'labels_train.txt'
docx_train = load_doc(in_filename_tweets)
docsent_train=load_doc(in_filename_sent)
docy_train = load_doc(in_filename_emoji)
tweets= docx_train.split('\n')
sentiment = docsent_train.split('\n')
emoji= docy_train.split('\n')

# declare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(tweets)
# equalize inputs
standard=count_max(sequences)
X=sq.pad_sequences(sequences, padding='post', maxlen=standard)
unique_words=len(np.unique(X))
word_index=tokenizer.word_index
sent=[]
print(sentiment[1])
for items in sentiment:
  indices=[int(items[0]),int(items[1]), int(items[2])]
  sent.append(indices)
X_sent=sq.pad_sequences(sent, padding='post', maxlen=standard)
print(X_sent[0])
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

# prepare embedding matrix
num_words = min(unique_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words with no embed index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# declare model
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=standard*2,
                            trainable=False)
sequence_input = Input(shape=(standard,), dtype='int32')
sent_input = Input(shape=(dim_sent,), dtype='int32')
merged = concatenate([sequence_input, sent_input])
embedded_sequences = embedding_layer(merged)

x = Bidirectional(LSTM(128, recurrent_dropout=0.4, return_sequences=True))(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.25)(x)
preds = Dense(20, activation='softmax')(x)

model = Model([sequence_input, sent_input], preds)
# compile and print summary of model

model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['categorical_accuracy'])
print(model.summary())
plot_model(model, 'emosent_pred_diagram.png', show_shapes=True)
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

# fit and train
for train, test in kfold.split(X_train):
	model.fit([X_train[train], X_sent_train[train]] ,y_train[train], epochs=1, 
  batch_size=size_batch, verbose=1, validation_data=([X_train[test],X_sent_train[test]], y_train[test]),
  callbacks=[reduce_lr, model_checkpoint_callback])
# save the model to file
model_json = model.to_json()
with open("model_sentemo.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_sentemo.h5")
print("Saved model to disk")

# save the tokenizer
dump(tokenizer, open('tokenizer_sentemo.pkl', 'wb'))

#load json and create model
json_file = open('model_sentemo.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_sentemo.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
score = loaded_model.evaluate([X_test, X_sent_test], Y_test, verbose=1)
print("Accuracy:" + str(score[1]*100) + "%")
print("Loss:" + str(score[0]))