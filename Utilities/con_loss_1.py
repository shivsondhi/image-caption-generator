import keras.backend as K
import tensorflow as tf
tf.set_random_seed(1)

def con_loss_fn_new(y_true, y_pred):
	y_ref = y_pred[:,0:28510]
	y_tar = y_pred[:,28510:]
	y_real = y_true[:,0:28510]

	dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_real,y_tar))))

	step_1 = tf.log(y_tar) - tf.log(y_ref)
	step_2 = tf.negative(step_1)
	step_3 = tf.exp(step_2)
	step_4 = tf.add(1.0, step_3)
	step_5 = tf.div(1.0, step_4)
	step_6 = tf.log(step_5)
	step_7_a = tf.subtract(1.0, step_5)
	step_7_b = tf.log(step_7_a)
	step_8 = tf.multiply(dist, step_7_b)
	loss = tf.abs(tf.reduce_mean(tf.add(step_6, step_8)))
	return loss