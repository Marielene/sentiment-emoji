from sklearn.feature_extraction.text import CountVectorizer
from numpy import loadtxt
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
# load the dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import hashing_trick
from nltk.tokenize import word_tokenize
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# load prepped data
in_filenameX = 'processed.txt'
in_filenameY = 'processedEmoji.txt'
docTweets = load_doc(in_filenameX)
docEmojis = load_doc(in_filenameY)
numbers= docEmojis.split('\n')
lines = docTweets.split('\n')


# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(lines)
print(sequences[1])
# standardize input
standard=0
for sequence in sequences:
	if(standard<len(sequence)):
		standard=len(sequence)
for sequence in sequences:
	if(len(sequence)<standard):
		for i in range(0,standard):
			sequence.append(0)
#for i in range(0 ,len(sequences)):
#	sequences[i].append(numbers[])
X = array(sequences)
#print(X[1])
y=array(numbers)
# define the keras model
model = Sequential()
model.add(Dense(128, input_dim=standard, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
# compile model
# fit model
model.fit(X, y, batch_size=128, epochs=20)
print(model.summary())
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))