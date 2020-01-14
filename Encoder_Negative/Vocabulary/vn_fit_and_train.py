from pickle import load, dump


def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

#create a set of all image_ids in the dataset 
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split("\n"):
		if len(line) < 1:
			continue
		identifier = line.split(' ')[0]
		dataset.append(identifier)
	return set(dataset)

#create a dictionary with image_id as key and a list of relevant captions
#as the value. Make sure to add startseq and endseq for every caption.
def load_clean_desc(filename, dataset):
	i = 0
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split():
		tokens = line.split("\n")
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = "startseq" + " ".join(image_desc) + "endseq"
			descriptions[image_id].append(desc)
			i += 1
			print(i)
	return descriptions

#pickle file is a dictionary
def load_photo_features(filename, dataset):
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features

#BUCK UP YOU SONOFABITCH 