import random
import json

def load_negative_desc():
	#train is the set of image_ids and desc is the dictionary of id-captions
	with open("training_descriptions.json", 'r') as json_file:
		positive_desc = json.load(json_file)
	
	negative_desc = {}
	l = list(positive_desc.keys())
	for x in positive_desc:
		negative_desc[x] = list()
		for _ in range(5):
			r = random.choice(l)
			while r==x:
				r = random.choice(l)
			negative_desc[x].append(positive_desc[r][random.randint(0,4)])
	
	with open("negative_descriptions.json", 'w') as json_file:
		json.dump(negative_desc, json_file)

	return("Done!")

def generate_negative_pairs(train, desc):
	#train is the set of image_ids from vocab.py and desc is the dictionary of id-captions
	size = len(train)

	negative_desc = {}
	for x in train:
		visited = list()
		visited.append(x)
		negative_descriptions[x] = list()
		for i in range(5):
			r = random.randint(1,size)
			if train[r] not in visited:
				negative_descriptions[x].append(desc[r])
				visited.add(r)
			else:
				i -= 1

if __name__ == "__main__":
	#print(load_negative_desc())
	with open("negative_descriptions.json") as json_file:
		n = json.load(json_file)

	with open("training_descriptions.json") as json_file:
		p = json.load(json_file)

		print(n==p)