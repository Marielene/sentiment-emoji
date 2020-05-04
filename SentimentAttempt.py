import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D, GRU, Embedding
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
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
StopW=list(stopwords.words('english'))
StopW.append('ive')
StopW.append('youve')
StopW.append('im')
StopW.append('user')
 # Load data
cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
data = data[['text','sentiment']]
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z\s]','',x)))
data['text'] = data['text'].apply(lambda x: x.lower())
data['text']= data['text'].apply((lambda x: lemmatizer.lemmatize(x)))
#for word in StopW:
#  data['text']= data['text'].apply((lambda x: re.sub(r"\b" + word + r"\b", "",x)))
for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
#Prepare data
#max_features = 10000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
standard=count_max(X)
vocab_size = len(tokenizer.word_index) + 1
X = pad_sequences(X, maxlen=standard)

Y = pd.get_dummies(data['sentiment'].values)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)



# Validate model
validation_size = 1500
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

input_dim = X_train.shape[0]  # Number of features

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=standard))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
print(model.summary())
 # fit model
model.fit(X_train, Y_train, epochs=10, shuffle=True, verbose=1, validation_split=0.001, batch_size=128)
#print(model.summary())
#Evaluate Model
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
dump(tokenizer, open('tokenizerSent.pkl', 'wb'))