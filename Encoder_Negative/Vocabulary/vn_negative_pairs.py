import random
import json
from pprint import pprint 
import string
import vn_fit_and_train
import vn_tokens

#To create negative image-caption pairs for CL

def load_negative_desc(train, positive_desc):
	#train is the set of image_ids and desc is the dictionary of id-captions
	negative_desc = {}
	size = len(train)
	for x in train:
		visited = []
		visited.append(x)
		negative_desc[x] = list()
		for i in range(5):
			r = random.randint(1,size)
			if train[r] not in visited:
				negative_desc[x].append(positive_desc[r][i])
				visited.add(r)
			else:
				i -= 1
	return negative_desc


if __name__ == "__main__":
	with open('F:\\Datasets\\MSCOCO\\annotations_trainval2017\\annotations\\captions_train2017.json') as json_file:
		text = json.load(json_file)
	#text is a dictionary with one key and a list of dictionaries as the value

	mapping = {}
	for a in text['annotations']:
		img_id = a['image_id']
		img_desc = a['caption']

		if img_id not in mapping:
			mapping[img_id] = list()
		mapping[img_id].append(img_desc)
	#mapping is a dictionary with lists as values

	#clean every caption by removing punctuation, lowercasing, less than one..
	#..character remove, remove words with numbers.
	
	translator = str.maketrans(dict.fromkeys(string.punctuation))
	#side note:
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
	#refer to machinelearningmastery
	all_desc = set()
	for key in mapping.keys():
		[all_desc.update(d.split()) for d in mapping[key]]
	#all_desc is a set of all characters learnt 

	print('Vocabulary size: %d' %len(all_desc))

	#shift cleaned key-image_caption pairs to new file
	lines = list()
	for key, desc_list in mapping.items():
		for desc in desc_list:
			lines.append(str(key) + ' ' + str(desc))

	data_p = '\n'.join(lines)

	filename_p = 'positive_captions.txt'
	file = open(filename_p, 'w')
	file.write(data_p)
	file.close()

#list of all image ids [id1, id2, id3...]
	train = list(vn_fit_and_train.load_set(filename_p))
	print("Done, " + str(len(train)))
	
#dictionary {image_id1: [caption1, caption2, caption3...], image_id2: [cap1, cap2, cap3]}
	true_descriptions = vn_fit_and_train.load_clean_desc('positive_captions.txt', train)
	print("Done, " + str(len(true_descriptions)))
	
	false_descriptions = load_negative_desc(train, true_descriptions)
	print("Done")

	lines = list()
	for key, desc_list in false_descriptions.items():
		for desc in desc_list:
			lines.append(str(key) + ' ' + str(desc))

	data_n = '\n'.join(lines)

	filename_n = 'negative_captions.txt'
	file = open(filename_n, 'w')
	file.write(data_n)
	file.close()

#dictionary {id1: features, id2: features, id3:features...}
	train_features = vn_fit_and_train.load_photo_features('features.pkl', train)
	print("Training photos: %d" %len(train_features))

#train_descriptions is a dictionary of image_ids to list of captions.
	tokenizer = vn_tokens.create_tokenizer(false_descriptions)
#creates dictionary mapping each word to a unique int value
	vocab_size = len(tokenizer.word_index) + 1
	print("Vocabulary Size: %d" %vocab_size)

	max_len = vn_tokens.max_length(false_descriptions)
	print ("Maximum caption length: %d" %max_len)
	X1_train, X2_train, y_train = vn_tokens.create_sequences(tokenizer, max_len, false_descriptions, train_features)
	print(train_features)
