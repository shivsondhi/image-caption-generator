from os import listdir
from pickle import dump
from keras.models import Model 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from inceptions import GoogleNet
import PIL
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

#import googlenet
'''
#extract features using googlenet model
def extract_features(directory):
	model = GoogleNet()

	model.layer.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

	print(model.summary())
'''

#define directory
directory = 'F:\\Datasets\\MSCOCO\\MSCOCO_train_2017\\train2017'

model = GoogleNet()

#model.layer.pop()
model = Model(inputs=model.inputs, outputs=model.output)

print(model.summary())
no=1

#create dictionary called features with image features
features = dict()
for name in listdir(directory):
	filename = str(directory + '/' + name)
	print(filename+" "+str(no))
	no += 1
	image = load_img(filename, target_size=(224,224))

	image = img_to_array(image)
#print(image.shape[0], image.shape[1], image.shape[2])
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

#figure out what happens in the preprocess_input() function
	image = preprocess_input(image)

	feature = model.predict(image, verbose=0)
	image_id = name.split('.')[0]

	features[image_id] = feature
	#break

print('Extracted features: %d' %len(features))

#save features to pickle file
dump(features, open('features.pkl', 'wb'))
