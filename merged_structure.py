import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Bidirectional, Input, Embedding, concatenate
from sklearn.model_selection import KFold
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing import sequence as sq
import re
import random
from keras.utils import plot_model
from keras.utils import to_categorical
from nltk.stem import WordNetLemmatizer
from keras.models import Model
from keras.models import model_from_json
from sklearn.model_selection import KFold
from pickle import dump

EMBEDDING_DIM=50
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

#load embedding data
print('Indexing word vectors.')

embeddings_index = {}
with open('glove.twitter.27B.50d.txt', encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# load emoji data
in_filenameX = 'processed.txt'
in_filenameY = 'processedEmoji.txt'
docx_train = load_doc(in_filenameX)
docy_train = load_doc(in_filenameY)
tweets= docx_train.split('\n')
emoji= docy_train.split('\n')

  # Vars to clean data
lemmatizer = WordNetLemmatizer();

 # Load sentiment data
cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
data = data[['text','sentiment']]
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z\s]','',x)))
data['text'] = data['text'].apply(lambda x: x.lower())
data['text']= data['text'].apply((lambda x: lemmatizer.lemmatize(x)))

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
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
size_batch=128
print(X_sent.shape)
print(X_emoji.shape)
#Prepare embedding matrix
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


#Sentiment embedding
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=standard,
                            trainable=True, mask_zero=True)
sequence_input_sent = Input(shape=(standard,), dtype='int32', name='sent_input')
sequence_input_emo = Input(shape=(standard,), dtype='int32', name='emo_input')
embedded_sequences_emo= embedding_layer(sequence_input_emo)
embedded_sequences_sent = embedding_layer(sequence_input_sent)


#skeleton for emoji
emo = Bidirectional(LSTM(128, recurrent_dropout=0.5, return_sequences=True, name='emo_bi_lstm'))(embedded_sequences_emo)
sent = Bidirectional(LSTM(64, recurrent_dropout=0.4, return_sequences=True, name='sent_bi_lstm'))(embedded_sequences_sent)
merged=concatenate([emo, sent])
x = LSTM(128, recurrent_dropout=0.4, return_sequences=True, name='standard_lstm2')(merged)
x = LSTM(64, recurrent_dropout=0.2, name='standard_lstm3')(x)
#sent = LSTM(32, recurrent_dropout=0.2)(sent)

#Outputs
preds_emoji = Dense(20, activation='softmax', name='emo_output')(x)
preds_sent = Dense(2, activation='softmax', name='sent_output')(x)

#MODEL CREATED JIJI GIRL
model = Model(inputs=[sequence_input_emo, sequence_input_sent], outputs=[preds_emoji, preds_sent])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.05),
    loss={'sent_output': keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="auto", name="sparse_categorical_crossentropy"),
          'emo_output': keras.losses.CategoricalCrossentropy(from_logits=True, reduction="auto", name="categorical_crossentropy")},
    metrics={'sent_output': keras.metrics.SparseCategoricalAccuracy(),
            'emo_output': keras.metrics.CategoricalAccuracy()})

#Summary & struct
print(model.summary())
plot_model(model, 'Abomination.png', show_shapes=True)
# crossvalidator
kfold=KFold(n_splits=4)
# Fit on dicts
for train, test in kfold.split(X_sent):
    model.fit([X_emoji[train], X_sent[train]],
            [Y_emoji[train], Y_sent[train]],
	          batch_size=size_batch,
            validation_data=([X_emoji[test], X_sent[test]], [Y_emoji[test], Y_sent[test]]),
	          epochs=1, shuffle=True)
#model.fit([X_emoji,X_sent], [Y_emoji, Y_sent], shuffle=True,
#           batch_size=size_batch, epochs=4, validation_split=0.1)
# save the model to file
model_json = model.to_json()
with open("model_Merged.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_Merged_weights.h5")
print("Saved model to disk")

# save the tokenizer
dump(tokenizer, open('combo_tokenizer.pkl', 'wb'))