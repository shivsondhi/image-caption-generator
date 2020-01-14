from numpy import argmax
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


#substitutes the correct word for the given integer value
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


def generate_desc(model, tokenizer, photo, max_length):
	print("Inside generate_desc")
	in_text = 'startseq'
	for i in range(max_length):
		#convert words to integer sequence and pad with zeros till max_length
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		#predict next word in the sequence	
		y_op = model.predict([photo,sequence], verbose=0)
		#slice to get target output
		yhat = y_op[:,0:28510]
		#integer with highest probability after softmax	
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	print("Inside evaluate_model")
	i = 1
	actual, predicted = list(), list()
	with open(descriptions, 'r') as json_file:
		for key, desc_list in json.load(json_file).items():
			yhat = generate_desc(model, tokenizer, photos[key], max_length)
			references = [d.split() for d in desc_list]
			actual.append(references)
			predicted.append(yhat.split())
			print(str(i))
			i += 1

		#print BLEU scores
		'''print('BLEU-1: {}'.format(corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))))
		print('BLEU-2: {}'.format(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))))
		print('BLEU-3: {}'.format(corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))))
		print('BLEU-4: {}'.format(corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))))'''

		print('BLEU-1: {}'.format()
		print('BLEU-2: {}'.format()
		print('BLEU-3: {}'.format()
		print('BLEU-4: {}'.format()
