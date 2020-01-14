from keras.preprocessing.text import Tokenizer
import numpy as np

#descriptions is a dictionary with image_id as key and a list of captions as value
#3 functions for the tokenizer

#dictionary of descs (train_descriptions) to list of descs
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

#fit tokenizer
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
#trains over the list of sentences provided	
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)


#photos and descriptions are a dictionary of lists
#photos has image features and descriptions has captions for every image_id
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	for key, desc_list in descriptions.items():
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

				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return np.array(X1), np.array(X2), np.array(y)


#X1 is the input with photo features
#X2 is the input with list of words generated till now
#y is the output with one-hot encoded, next predicted word

#BUCK UP YOU SONOFABITCH 