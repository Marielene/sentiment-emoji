import os
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import codecs
#Reduce data further by removing unnecessary wording
lemmatizer = WordNetLemmatizer() 
StopW=list(stopwords.words('english'))
StopW.append('ive')
StopW.append('youve')
StopW.append('im')
StopW.append('user')
def file_parser(filenaem, tweets):
	datafile = open(filenaem, encoding="utf8")
	for line in datafile:
		# LOWERCASE
		line=line.lower()
		#remove whitespace at ends
		line=line.strip()
		#ensure the tweets are stopword-free
		for word in StopW:
			checkstr= r"\b" + word + r"\b"
			line=re.sub(checkstr, '', line)
		line= lemmatizer.lemmatize(line)
		# single letters removal
		line = re.sub(r"\b[a-zA-Z]\b", '', line, flags=re.UNICODE)
		#ensure the tweet is only spaces and letters
		line = re.sub('[^a-zA-Z ]+', '', line, flags=re.UNICODE)
		#replace all whitespace with single space.
		line=re.sub(r"\s+", ' ', line, flags=re.UNICODE)
		#clear edges again to be sure
		line=line.strip()
		#print(line)
		tweets.append(line)
	datafile.close()
#separate function with no cleaning
def file_parser_emoji(filenaem, emojis):
	datafile=open(filenaem,"r")
	for line in datafile:
		line=re.sub(r"\n", "", line, flags=re.UNICODE)
		emojis.append(line)
	datafile.close()
# save tokens to file, one dialog per line
def save_file(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, "w+", encoding="utf8")
	file.write(data)
	file.close()
tweets=[]
file_parser("C:/Users/setsu/documents/attempts/emoji_prediction/emoji_prediction/train/us_train.text", tweets)
save_file(tweets, "processed.txt")
emojis=[]
#file_parser_emoji("C:/Users/setsu/documents/attempts/emoji_prediction/emoji_prediction/train/us_train.labels", emojis)
#save_file(emojis, "processedEmoji.txt")
print("Done!")