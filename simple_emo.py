import numpy as np
import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence as sq
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Activation, Dropout
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
from keras.models import model_from_json
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer	
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
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
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.01

# crossvalidator
kfold = KFold(n_splits=5)

#load embedding data
print('Indexing word vectors.')

embeddings_index = {}
with open('glove.6B.50d.txt', encoding='latin-1') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


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
X=sq.pad_sequences(sequences, padding='post', maxlen=standard)
unique_words=len(np.unique(X))
word_index=tokenizer.word_index
# organize data.
emoji_prep=prep_data_int(emoji)
Y=to_categorical(emoji_prep)
print(Y[0])
X_test=X[-1000:]
X_train=X[:-1000]
Y_test=Y[-1000:]
y_train=Y[:-1000]
size_batch=128

#Prepare embedding matrix
num_words = min(unique_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=standard,
                            trainable=False, mask_zero=True)
sequence_input = Input(shape=(standard,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Bidirectional(LSTM(128, recurrent_dropout=0.2, return_sequences=True))(embedded_sequences)
x = Bidirectional(LSTM(64, recurrent_dropout=0.2, return_sequences=True))(x)
x = Bidirectional(LSTM(32, recurrent_dropout=0.2))(x)
preds = Dense(20, activation='softmax')(x)

model = Model(sequence_input, preds)
# compile and print summary of model
model.compile(loss='categorical_crossentropy',
              optimizer='ADAM',
              metrics=['categorical_accuracy'])
print(model.summary())
#Fit and train
#wrapped_model = KerasClassifier(build_fn=model, epochs=5, batch_size=size_batch, verbose=1)
#results = cross_val_score(wrapped_model, X_train, y_train, cv=kfold)
#print("Baseline: %s %  (%s)" % (results.mean()*100, results.std()*100))
for train, test in kfold.split(X_train, y_train):
	model.fit(X_train[train],y_train[train], epochs=1, batch_size=size_batch, verbose=1, validation_data=(X_train[test], y_train[test]))
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
score = loaded_model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy:" + str(score[1]*100) + "%")
print("Loss:" + str(score[0]))