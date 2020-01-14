import vp_fit_and_train
import vp_tokens	
import gc 
from dp_decoder import define_model
from keras import backend as K
from keras.models import load_model

if __name__ == "__main__":
	#set of all image ids [id1, id2, id3...]
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
	
	model = define_model(vocab_size, max_len)
	print(x)
	model = load_model('model_{}.h5'.format('7'))
	#specifiy new learning rate
	K.set_value(model.optimizer.lr, 0.0004)
	epochs = 11
	steps = 118286
	for i in range(8,epochs):
		#generator = vp_tokens.data_generator(tokenizer, max_len, vocab_size, train_features)
		model.fit_generator(vp_tokens.data_generator("training_descriptions.json", tokenizer, max_len, vocab_size, train_features), epochs=1, steps_per_epoch=steps, verbose=1, use_multiprocessing=False, max_queue_size=2)
		model.save('model_{}.h5'.format(str(i)))
