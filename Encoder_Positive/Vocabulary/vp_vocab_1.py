import json
from pprint import pprint 
import string
import vp_fit_and_train
import vp_tokens

if __name__ == "__main__":
	#move contents of json file to a variable text
	with open('D:\\Datasets\\MSCOCO\\annotations_trainval2017\\annotations\\captions_train2017.json') as json_file:
		text = json.load(json_file)
	#text is a dictionary with one key and a list of dictionaries as the value

	'''create dictionary, with image_id as key and list of caption_ids 
	as values'''

	'''
	alternate:
	img_id = image_id
	img_desc = caption

	if img_id not in mapping:
		mapping[img_id] = list()
	mapping[img_id].append(img_desc)
	'''
	mapping = {}
	for a in text['annotations']:
		img_id = a['image_id']
		img_desc = a['caption']

		img_id = str(img_id).zfill(12)
		
		if img_id not in mapping:
			mapping[img_id] = list()
		mapping[img_id].append(img_desc)
	#mapping is a dictionary with lists as values

	'''clean every caption by removing punctuation, lowercasing, less than one
	character and words with numbers.'''

	translator = str.maketrans(dict.fromkeys(string.punctuation))
	#alternate:
	#translator = str.maketrans({key: None for key in string.punctuation + 'abc'})

	for key, val in mapping.items():	#key is id and val is list
		for i in range(len(val)):		#i iterates through len(val) [list]
			desc = val[i]				#desc = each element in val  [list]
			desc = desc.split()			#desc is now list of chars
			desc = [word.lower() for word in desc]
			desc = [w.translate(translator) for w in desc]
			desc = [word for word in desc if len(word)>1]
			desc = [word for word in desc if word.isalpha()]
			#desc is now still a list

			val[i] = ' '.join(desc)	#joins eles in desc with '' in between

	#shift all words in all captions to a set using set update and for loops
	all_desc = set()
	for key in mapping.keys():
		[all_desc.update(d.split()) for d in mapping[key]]
	#all_desc is a set of all characters learnt 

	print('Vocabulary size: %d' %len(all_desc))

	#print(len(mapping))
	#Because image file was corrupt
	del mapping['000000424797']
	#print(len(mapping))

	#shift cleaned key-image_caption pairs to new file
	lines = list()
	for key, desc_list in mapping.items():
		for desc in desc_list:
			lines.append(str(key) + ' ' + str(desc))

	data = '\n'.join(lines)

	filename = 'id_captions.txt'
	file = open(filename, 'w')
	file.write(data)
	file.close()

	#print("End temp")
	
	#set of all image ids [id1, id2, id3...]
	train = vp_fit_and_train.load_set('id_captions.txt')
	print("Size of dataset (Number of image id's): %d" %len(train))

	#dictionary {image_id1: [caption1, caption2, caption3...], image_id2: [cap1, cap2, cap3]}
	train_descriptions = vp_fit_and_train.load_clean_desc('id_captions.txt', train)
	print("Training descriptions: %d" %len(train_descriptions))


	#dictionary {id1: features, id2: features, id3:features...}
	train_features = vp_fit_and_train.load_photo_features('all_features.pkl', train)
	print("Training photos: %d" %len(train_features))

	#train_descriptions is a dictionary of image_ids to list of captions.
	tokenizer = vp_tokens.create_tokenizer(train_descriptions) #train_descriptions

	#creates dictionary mapping each word to a unique int value
	vocab_size = len(tokenizer.word_index) + 1
	print("Vocabulary Size: %d" %vocab_size)

	max_len = vp_tokens.max_length(train_descriptions)
	print ("Maximum caption length: %d" %max_len)

	X1_train, X2_train, y_train = vp_tokens.create_sequences(tokenizer, max_len, train_descriptions, train_features, vocab_size)
