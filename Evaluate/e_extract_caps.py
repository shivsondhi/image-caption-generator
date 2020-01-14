import json
from pprint import pprint 
import string
import vp_fit_and_train
import vp_tokens

if __name__ == "__main__":
	#move contents of json file to a variable text
	with open('D:\\Datasets\\MSCOCO\\annotations_trainval2017\\annotations\\captions_val2017.json', 'r') as json_file:
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

	#shift cleaned key-image_caption pairs to new file
	lines = list()
	for key, desc_list in mapping.items():
		for desc in desc_list:
			lines.append(str(key) + ' ' + str(desc))

	#This is where the newline is introduced x_x
	data = '\n'.join(lines)

	filename = 'validation_id_captions.txt'
	file = open(filename, 'w')
	file.write(data)
	file.close()

	print("End temp")
