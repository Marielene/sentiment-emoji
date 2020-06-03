import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding, Input
from sklearn.model_selection import KFold
from keras.models import Model
from keras.initializers import Constant
from keras.utils import plot_model
import re
from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
from pickle import dump
#Beauty, grace
def count_max(count_list):
	maxboi=0
	for item in count_list:
		if(maxboi<len(item)):
			maxboi=len(item)
	return maxboi
  # Vars to clean data
lemmatizer = WordNetLemmatizer();
 # load and process data.
cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
data = data[['text','sentiment']]
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z\s]','',x)))
data['text'] = data['text'].apply(lambda x: x.lower())
data['text']= data['text'].apply((lambda x: lemmatizer.lemmatize(x)))
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')


#Tokenize data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
standard=count_max(X)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index

# prepare embeddings, crossvalidator
EMBEDDING_DIM = 50
kfold = KFold(n_splits=4)

# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors.')

embeddings_index = {}
with open('glove.twitter.27B.50d.txt', encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# set X and Y
X = pad_sequences(X, maxlen=standard)
Y = np.array(data['sentiment'].values)
# change data tp fit with loss function
Y = np.where(Y==2, 1, Y)
Y = np.where(Y==4, 2, Y)
# prepare embedding matrix
num_words =  len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=standard,
                            trainable=False, mask_zero=True)
sequence_input = Input(shape=(standard,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Bidirectional(LSTM(64, recurrent_dropout=0.4, return_sequences=True))(embedded_sequences)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(32, recurrent_dropout=0.2))(x)
preds = Dense(3, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['sparse_categorical_accuracy'])
print(model.summary())
plot_model(model, 'sent_only_model.png', show_shapes=True)
for train, test in kfold.split(X,Y):
	model.fit(X[train], Y[train],
	          batch_size=64,
	          epochs=1,
	          validation_data=(X[test], Y[test]), shuffle=True)
model_json = model.to_json()
with open("modelSent.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelSent.h5")
print("Saved model to disk")

# save the tokenizer
dump(tokenizer, open('tokenizerSent.pkl', 'wb'))
#score = model.evaluate(X_final, Y_final, verbose=1)
#print("Accuracy:" + str(score[1]*100) + "%")
#print("Loss:" + str(score[0]))