from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, merge, Reshape, Activation, ZeroPadding2D
from keras.models import Model
from supporting_classes import LRN, CustomPad 
from keras.layers.merge import Concatenate
from keras import layers

#data steps
def GoogleNet(weights=None):
#stem starts
	inputs = Input(shape=(3,224,224))
	conv1 = Convolution2D(64, (7,7), strides=(2,2), padding='same', activation='relu', name='conv1')(inputs)

#print(conv1.shape)
	conv1_zero_pad = ZeroPadding2D(padding=(1,1))(conv1)
	custom_pad1 = CustomPad()(conv1_zero_pad)
	pool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(custom_pad1)

#print(pool1.shape)
	norm1 = LRN(name='norm1')(pool1)

#print(norm1.shape)
	conv2_red = Convolution2D(64, (1,1), padding='same', activation='relu', name='conv2_red')(norm1)
	conv2 = Convolution2D(192, (3,3), padding='same', activation='relu', name='conv2')(conv2_red)
	norm2 = LRN(name='norm2')(conv2)

#print(norm2.shape)
	norm2_zero_pad = ZeroPadding2D(padding=(1,1))(norm2)
	custom_pad2 = CustomPad()(norm2_zero_pad)
	pool2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool2')(custom_pad2)
#print(pool2.shape)

#inception modules start

#inception -----------------------------------------------------------3a
	t_1_3a = Convolution2D(64, (1,1), padding='same', activation='relu', name='t_1_3a')(pool2)

	t_2_3a = Convolution2D(64, (1,1), padding='same', activation='relu')(pool2)
	t_2_3a = Convolution2D(64, (3,3), padding='same', activation='relu')(t_2_3a)

	t_3_3a = Convolution2D(64, (1,1), padding='same', activation='relu')(pool2)
	t_3_3a = Convolution2D(96, (3,3), padding='same', activation='relu')(t_3_3a)
	t_3_3a = Convolution2D(96, (3,3), padding='same', activation='relu')(t_3_3a)
#print(t_3.shape)

	t_4_3a = AveragePooling2D((3,3), strides=1, padding='same')(pool2)
#print(t_4.shape)
	t_4_3a = Convolution2D(32, (1,1), padding='same', activation='relu')(t_4_3a)
	
	#inception3a = Concatenate([t_1, t_2, t_3, t_4], axis=1, name='inception3a')
	inception3a = merge([t_1_3a, t_2_3a, t_3_3a, t_4_3a], mode='concat', concat_axis=1, name='inception3a')

#inception------------------------------------------------------------3b
	t_1_3b = Convolution2D(64, (1,1), padding='same', activation='relu', name='t_1_3b')(inception3a)

	t_2_3b = Convolution2D(64, (1,1), padding='same', activation='relu')(inception3a)
	t_2_3b = Convolution2D(96, (3,3), padding='same', activation='relu')(t_2_3b)

	t_3_3b = Convolution2D(64, (1,1), padding='same', activation='relu')(inception3a)
	t_3_3b = Convolution2D(96, (3,3), padding='same', activation='relu')(t_3_3b)
	t_3_3b = Convolution2D(96, (3,3), padding='same', activation='relu')(t_3_3b)

	t_4_3b = AveragePooling2D((3,3), strides=1, padding='same')(inception3a)	
	t_4_3b = Convolution2D(64, (1,1), padding='same', activation='relu')(t_4_3b)

	inception3b = merge([t_1_3b, t_2_3b, t_3_3b, t_4_3b], mode='concat', concat_axis=1, name='inception3b')

#inception------------------------------------------------------------3c
	t_1_3c = Convolution2D(0, (1,1), padding='same', activation='relu', name='t_1_3c')(inception3b)

	t_2_3c = Convolution2D(128, (1,1), padding='same', activation='relu')(inception3b)
	t_2_3c = Convolution2D(160, (3,3), padding='same', activation='relu')(t_2_3c)

	t_3_3c = Convolution2D(64, (1,1), padding='same', activation='relu')(inception3b)
	t_3_3c = Convolution2D(96, (3,3), padding='same', activation='relu')(t_3_3c)
	t_3_3c = Convolution2D(96, (3,3), padding='same', activation='relu')(t_3_3c)

	t_4_3c = MaxPooling2D((3,3), strides=1, padding='same')(inception3b)

	inception3c = merge([t_1_3c, t_2_3c, t_3_3c, t_4_3c], mode='concat', concat_axis=1, name='inception3c')

#inception------------------------------------------------------------4a
	t_1 = Convolution2D(224, (1,1), padding='same', activation='relu', name='t_1_4a')(inception3c)

	t_2 = Convolution2D(64, (1,1), padding='same', activation='relu')(inception3c)
	t_2 = Convolution2D(96, (3,3), padding='same', activation='relu')(t_2)

	t_3 = Convolution2D(96, (1,1), padding='same', activation='relu')(inception3c)
	t_3 = Convolution2D(128, (3,3), padding='same', activation='relu')(t_3)
	t_3 = Convolution2D(128, (3,3), padding='same', activation='relu')(t_3)

	t_4 = AveragePooling2D((3,3), strides=1, padding='same')(inception3c)
	t_4 = Convolution2D(128, (1,1), padding='same', activation='relu')(t_4)

	inception4a = merge([t_1, t_2, t_3, t_4], mode='concat', concat_axis=1, name='inception4a')

#inception------------------------------------------------------------4b
	t_1 = Convolution2D(192, (1,1), padding='same', activation='relu', name='t_1_4b')(inception4a)

	t_2 = Convolution2D(96, (1,1), padding='same', activation='relu')(inception4a)
	t_2 = Convolution2D(128, (3,3), padding='same', activation='relu')(t_2)

	t_3 = Convolution2D(96, (1,1), padding='same', activation='relu')(inception4a)
	t_3 = Convolution2D(128, (3,3), padding='same', activation='relu')(t_3)
	t_3 = Convolution2D(128, (3,3), padding='same', activation='relu')(t_3)

	t_4 = AveragePooling2D((3,3), strides=1, padding='same')(inception4a)
	t_4 = Convolution2D(128, (1,1), padding='same', activation='relu')(t_4)

	inception4b = merge([t_1, t_2, t_3, t_4], mode='concat', concat_axis=1, name='inception4b')

#inception------------------------------------------------------------4c
	t_1 = Convolution2D(160, (1,1), padding='same', activation='relu', name='t_1_4c')(inception4b)

	t_2 = Convolution2D(128, (1,1), padding='same', activation='relu')(inception4b)
	t_2 = Convolution2D(160, (3,3), padding='same', activation='relu')(t_2)

	t_3 = Convolution2D(128, (1,1), padding='same', activation='relu')(inception4b)
	t_3 = Convolution2D(160, (3,3), padding='same', activation='relu')(t_3)
	t_3 = Convolution2D(160, (3,3), padding='same', activation='relu')(t_3)

	t_4 = AveragePooling2D((3,3), strides=1, padding='same')(inception4b)
	t_4 = Convolution2D(128, (1,1), padding='same', activation='relu')(t_4)

	inception4c = merge([t_1, t_2, t_3, t_4], mode='concat', concat_axis=1, name='inception4c')

#inception------------------------------------------------------------4d
	t_1 = Convolution2D(96, (1,1), padding='same', activation='relu', name='t_1_4d')(inception4c)

	t_2 = Convolution2D(128, (1,1), padding='same', activation='relu')(inception4c)
	t_2 = Convolution2D(192, (3,3), padding='same', activation='relu')(t_2)

	t_3 = Convolution2D(160, (1,1), padding='same', activation='relu')(inception4c)
	t_3 = Convolution2D(192, (3,3), padding='same', activation='relu')(t_3)
	t_3 = Convolution2D(192, (3,3), padding='same', activation='relu')(t_3)

	t_4 = AveragePooling2D((3,3), strides=1, padding='same')(inception4c)
	t_4 = Convolution2D(128, (1,1), padding='same', activation='relu')(t_4)

	inception4d = merge([t_1, t_2, t_3, t_4], mode='concat', concat_axis=1, name='inception4d')

#inception------------------------------------------------------------4e
	t_1 = Convolution2D(0, (1,1), padding='same', activation='relu', name='t_1_4e')(inception4d)

	t_2 = Convolution2D(128, (1,1), padding='same', activation='relu')(inception4d)
	t_2 = Convolution2D(192, (3,3), padding='same', activation='relu')(t_2)

	t_3 = Convolution2D(192, (1,1), padding='same', activation='relu')(inception4d)
	t_3 = Convolution2D(256, (3,3), padding='same', activation='relu')(t_3)
	t_3 = Convolution2D(256, (3,3), padding='same', activation='relu')(t_3)

	t_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception4d)

	inception4e = merge([t_1, t_2, t_3, t_4], mode='concat', concat_axis=1, name='inception4e')

#inception------------------------------------------------------------5a
	t_1 = Convolution2D(352, (1,1), padding='same', activation='relu', name='t_1_5a')(inception4e)

	t_2 = Convolution2D(192, (1,1), padding='same', activation='relu')(inception4e)
	t_2 = Convolution2D(320, (3,3), padding='same', activation='relu')(t_2)

	t_3 = Convolution2D(160, (1,1), padding='same', activation='relu')(inception4e)
	t_3 = Convolution2D(224, (3,3), padding='same', activation='relu')(t_3)
	t_3 = Convolution2D(224, (3,3), padding='same', activation='relu')(t_3)

	t_4 = AveragePooling2D((3,3), strides=1, padding='same')(inception4e)
	t_4 = Convolution2D(128, (1,1), padding='same', activation='relu')(t_4)

	inception5a = merge([t_1, t_2, t_3, t_4], mode='concat', concat_axis=1, name='inception5a')

#inception------------------------------------------------------------5b
	t_1 = Convolution2D(352, (1,1), padding='same', activation='relu', name='t_1_5b')(inception5a)

	t_2 = Convolution2D(192, (1,1), padding='same', activation='relu')(inception5a)
	t_2 = Convolution2D(320, (3,3), padding='same', activation='relu')(t_2)

	t_3 = Convolution2D(192, (1,1), padding='same', activation='relu')(inception5a)
	t_3 = Convolution2D(224, (3,3), padding='same', activation='relu')(t_3)
	t_3 = Convolution2D(224, (3,3), padding='same', activation='relu')(t_3)

	t_4 = AveragePooling2D((3,3), strides=1, padding='same')(inception5a)
	t_4 = Convolution2D(128, (1,1), padding='same', activation='relu')(t_4)

	inception5b = merge([t_1, t_2, t_3, t_4], mode='concat', concat_axis=1, name='inception5b')

#pooling3
	pool3 = AveragePooling2D((7,7), padding='same')(inception5b)

#flatten and drop	
	flat = Flatten()(pool3)
	drop = Dropout(0.4, name='drop')(flat)

#classifier
	classifier = Dense(1000, name='classifier')(drop)
	classifier_act = Activation('softmax')(classifier)

#end
	model = Model(input=inputs, output=[drop])

	if weights:
		model.load_weights(weights)

	return model 
