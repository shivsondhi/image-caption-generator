import json
import random
from pprint import pprint

if __name__ == "__main__":
	filename = "training_descriptions.json"

	with open(filename, 'r') as json_file:
		d = json.load(json_file)
		
	#pprint(d)
	
	keys = list(d.keys())
	random.shuffle(keys)
	
	d_new = [(key, d[key]) for key in keys]
	
	for a, b in d_new:
		print(a, b)
	
	'''
	with open(filename, 'w') as json_file:
		json.dump(d_new, json_file)
	'''