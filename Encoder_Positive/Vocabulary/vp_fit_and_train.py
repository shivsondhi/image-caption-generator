import pickle
import gc
import json
import numpy as np

def load_doc(filename):
	file = open(filename, 'r')
	while True:
		text = file.readline()
		if not text:
			file.close()
			break
		yield text

#create a set of all image_ids in the dataset 
def load_set(filename):
	dataset = list()
	for line in load_doc(filename):
		if len(line) < 1:
			continue
		identifier = line.split(' ')[0]
		dataset.append(identifier)
	gc.collect()
	return set(dataset)

#create a dictionary with image_id as key and a list of relevant captions
#as the value. Make sure to add startseq and endseq for every caption.
def load_clean_desc(filename, dataset, out_file):
	descriptions = dict()
	for line in load_doc(filename):
		image_id, image_desc = line[0:12], line[12:]
		#print(image_id)
		#print(image_desc)
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = "startseq {} endseq".format("".join(image_desc))
			#print(desc)
			descriptions[image_id].append(desc)
	train_descs = len(descriptions)
	with open(out_file, 'w') as json_file:
		json.dump(descriptions, json_file)
	del descriptions
	gc.collect()
	print("Created JSON!")
	return train_descs

#pickle file is a dictionary
def load_photo_features(filename, dataset):
	all_features = {}
	with (open(filename, "rb")) as openfile:
		while True:
			try:
				all_features.update(pickle.load(openfile))
				print("Appended")
			except EOFError:
				break
	print("happening...")
	features = {k: all_features[k] for k in dataset}
	'''del all_features
	gc.collect()
	del features
	gc.collect()'''
	print("leaving!")
	return features

#BUCK UP YOU SONOFABITCH 