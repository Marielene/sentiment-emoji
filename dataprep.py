#HOW PARSE EMOJI ????? HOW LIVE???? WHEN FIND OUT????
import os
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
import codecs
#Reduce data further by removing unnecessary wording
StopW=set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer();
def filereaderMainData(filenaem, tweets):
	datafile = open(filenaem, encoding="utf8")
	for line in datafile:
		#replace ALL spaces with single space in tweet
		line=re.sub(r"\s+", " ", line, flags=re.UNICODE)
		#remove punctuation
		line=re.sub(r'[^\w\s]','',line.lower());
		line=lemmatizer.lemmatize(line);
		temp=line.split()
		temp=[word for word in temp if word not in StopW]
		line=' '.join(temp)
		tweets.append(line)
		#print(processed)
		#print(line)
	datafile.close()
#separate function with no cleaning
def fileReaderEmoji(filenaem, emojis):
	datafile=open(filenaem,"r")
	for line in datafile:
		line=re.sub(r"\n", "", line, flags=re.UNICODE)
		emojis.append(line)
	datafile.close()
# save tokens to file, one dialog per line
def saveProcessed(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, "w+", encoding="utf8")
	file.write(data)
	file.close()
tweets=[]
filereaderMainData("C:/Users/setsu/documents/attempts/emoji_prediction/emoji_prediction/train/us_train.text", tweets)
f= open("processed.txt","w+")
f.close()
saveProcessed(tweets, "processed.txt")
emojis=[]
f= open("processedEmoji.txt","w+")
f.close()
fileReaderEmoji("C:/Users/setsu/documents/attempts/emoji_prediction/emoji_prediction/train/us_train.labels", emojis)
saveProcessed(emojis, "processedEmoji.txt")
#Split in letters
#Organise as n-grams
#Set n-grams max val