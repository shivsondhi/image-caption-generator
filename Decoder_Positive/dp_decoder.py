from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add, concatenate
from keras.models import Model
import numpy as np
from keras.utils import plot_model
from keras import optimizers

#define the model
def define_model(vocab_size, max_length):
	#input is X1, all photo features
	inputs1 = Input(shape=(16384,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)

	#input is X2, all words generated till now, ignoring the padded 0's
	inputs2 = Input(shape=(max_length,))
	#embeds with input size=vocab_size, number of dimensions of vector space to map to=256
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256, return_sequences=True)(se2)

	#simple addition of input tensors
	decoder1 = add([fe2, se3])
	decoder2 = LSTM(256)(decoder1)
	#softmax over the entire vocabulary
	outputs = Dense(vocab_size, activation='relu')(decoder2)

	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	model.compile(loss='categorical_crossentropy', optimizer=adam)

	print(model.summary())
	plot_model(model, to_file='model_final.png', show_shapes=True)

	return model

'''
#fit model
if __name__ == "__main__":
	model = define_model(vocab_size, max_len)
	filepth = 'model-ep{epoch:03d}-loss{loss: .3f}-val_loss{val_loss: .3f}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

	model.fit([np.X1_train, np.X2_train], np.y_train, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([np.X1_test, np.X2_test], np.y_test))
'''