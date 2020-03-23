import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils.np_utils import to_categorical
import re
from keras.models import model_from_json
from pickle import dump
 # Load data
cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
data = data[['text','sentiment']]
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z\s]','',x)))
data['text'] = data['text'].apply(lambda x: x.lower())
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
#Prepare data
#max_fatures = 2000
#tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#tokenizer.fit_on_texts(data['text'].values)
#X = tokenizer.texts_to_sequences(data['text'].values)
#X = pad_sequences(X)

Y = pd.get_dummies(data['sentiment'].values)
vectorizer = CountVectorizer()
vectorizer.fit(data['text'].values)
X = vectorizer.transform(data['text'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
 # fit model
batch_size = 32

# Validate model
validation_size = 1500
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(Dense(100, input_dim=input_dim, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, epochs=10, verbose=1, validation_data=(X_validate, Y_validate), batch_size=32)
score = model.evaluate(X_test,Y_test)
print("Accuracy:", score)
# save the model to file
model_json = model.to_json()
with open("modelSent.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelSent.h5")
print("Saved model to disk")

# save the tokenizer
#dump(tokenizer, open('tokenizerSent.pkl', 'wb'))