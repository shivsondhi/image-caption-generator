from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import gc
import json
import random

#descriptions is a dictionary with image_id as key and a list of captions as value
#3 functions for the tokenizer

#dictionary of descs (train_descriptions) to list of descs
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

#fit tokenizer
def create_tokenizer(filename):
	with open(filename, 'r') as json_file:
		lines = to_lines(json.load(json_file))
	tokenizer = Tokenizer()
	#trains over the list of sentences provided	
	tokenizer.fit_on_texts(lines)
	del lines
	gc.collect()
	return tokenizer

def max_length(filename):
	with open(filename, 'r') as json_file:
		lines = to_lines(json.load(json_file))
	#print(x.count(' '))
	return max([x.count(' ') for x in lines]) 


#photos and descriptions are a dictionary of lists
#photos has image features and descriptions has captions for every image_id
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
	X1, X2, y = list(), list(), list()
	#print("Entered")
	for desc in desc_list:
	#seq equals the encoded caption, desc. 
		seq = tokenizer.texts_to_sequences([desc])[0]
		for i in range(1, len(seq)):
	#in_seq is a list of all words till i and out_seq is the next word
			in_seq, out_seq = seq[:i], seq[i]
	#pad in_seq with zero till all are uniform length
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
	#change out_seq to a one-hot vector
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	#print("...")
	del in_seq
	del out_seq
	gc.collect()
	#print("Exited")
	return np.array(X1), np.array(X2), np.array(y)

def data_generator(filename, tokenizer, max_len, vocab_size, photos):
	i = 0
	with open(filename, 'r') as json_file:
		#read file iteratively, in while loop
		#for key, desc_list in json.load(json_file).items():
		d = json.load(json_file)
	keys = list(d.keys())
	random.shuffle(keys)
	d_1 = [(key, d[key]) for key in keys]
	while True:
		for key, desc_list in d_1:
			#print(str(key))
			#extract the photo feature from the pickle file 
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_len, desc_list, photo, vocab_size)
			yield[[in_img, in_seq], out_word]

#X1 is the input with photo features
#X2 is the input with list of words generated till now
#y is the output with one-hot encoded, next predicted word

#BUCK UP YOU SONOFABITCH 