from keras.layers.core import Layer
from keras import backend as K
import tensorflow as tf 
import numpy as np

#Making PoolHelper a subclass of the Layer class in Keras.
class CustomPad(Layer):							#Subclass(Parent_class):
	def __init__(self, **kwargs):
		super(CustomPad, self).__init__(**kwargs)

	def call(self, x, mask=None):
		#Removing 0 padding from before first row and first column, all other ip dims remain same.
		return x[:,:,1:,1:]

	def get_config(self):
		config = {}
		base_config = super(CustomPad, self).getconfig()
		return dict(list(base_config.items()) + list(config.items()))

class LRN(Layer):
	def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
		self.alpha = alpha
		self.k = k
		self.beta = beta 
		self.n = n
		super(LRN, self).__init__(**kwargs)

	def call(self, x, mask=None):
		b, ch, r, c = x.shape
		p = 1
	#print(x.shape)
	#print(ch)

		'''b = x.get_shape().as_list()[0]
		
		b_tensor = K.shape(x)[0]
		sess = tf.Session()
		x_val = np.random.rand(1,56,63)
		b_val = sess.run(b_tensor, {x: x_val})
		print(x_val)'''

		half_n = self.n//2

		input_sqr = K.square(x)
	#print(input_sqr.shape)
		extra_channels = K.zeros((p, int(ch)+2*half_n, r, c))
		input_sqr = K.concatenate([extra_channels[:, :half_n, :, :], input_sqr, extra_channels[:, half_n + int(ch):, :, :]], axis=1)

		scale = self.k
		norm_alpha = self.alpha / self.n
		for i in range(self.n):
			scale += norm_alpha * input_sqr[:, i:i+int(ch), :, :]
		scale = scale **self.beta
		x = x / scale
		return x

	def get_config(self):
		config = {"alpha": self.alpha, "k": self.k, "beta": self.beta, "n": self.n}
		base_config = super(LRN, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))