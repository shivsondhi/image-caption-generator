
#substitutes the correct word for the given integer value
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):
	#convert words to integer sequence and pad with zeros till max_length
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)

	#predict next word in the sequence	
		yhat = model.predict([photo,sequence], verbose=0)
	#integer with highest probability after softmax	
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)

		if word is None:
			break

		in_text += ' ' + word

		if word == 'endseq':
			break

	return in_text

