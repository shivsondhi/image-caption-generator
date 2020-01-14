
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
			yield[[in_img, in_seq, in_img, in_seq], out_word]


if __name__ == "__main__":

	train = vp_fit_and_train.load_set('id_captions.txt')
	print("Size of dataset (Number of image id's): {}".format(len(train)))

	'''
	#dictionary {image_id1: [caption1, caption2, caption3...], image_id2: [cap1, cap2, cap3]}
	train_descs = vp_fit_and_train.load_clean_desc('id_captions.txt', train, 'training_descriptions.json')
	print("Training descriptions: {}".format(train_descs))
	'''
	
	#dictionary {id1: features, id2: features, id3:features...}
	train_features = vp_fit_and_train.load_photo_features('all_features.pkl', train)
	print("Training photos: {}".format(len(train_features)))

	del train
	gc.collect()

	#train_descriptions is a dictionary of image_ids to list of captions.
	tokenizer = vp_tokens.create_tokenizer("training_descriptions.json") #train_descriptions

	#creates dictionary mapping each word to a unique int value
	vocab_size = len(tokenizer.word_index) + 1
	print("Vocabulary Size: {}".format(vocab_size))

	max_len = vp_tokens.max_length("training_descriptions.json")
	print ("Maximum caption length: {}".format(max_len))

	gc.collect()
	
	model = cl_model(vocab_size, max_len)
	model = load_model('model_{}.h5'.format('3'))

	epochs = 11
	steps = 118287
	for i in range(4,epochs):
		#generator = vp_tokens.data_generator(tokenizer, max_len, vocab_size, train_features)
		model.fit_generator(data_generator("cl_training_descriptions.json", tokenizer, max_len, vocab_size, cl_train_features), epochs=1, steps_per_epoch=steps, verbose=1, use_multiprocessing=False, max_queue_size=2)
		model.save('model_{}.h5'.format(str(i)))
