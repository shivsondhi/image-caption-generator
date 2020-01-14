import and everything

def cl_model(vocab_size, max_len):

	in_seq_t = Input(shape=(49,))
	in_seq_r = Input(shape=(49,))
	inputs1_t = Input(shape=(16384,))
	inputs1_r = Input(shape=(16384,))

	model_tar = dp_decoder_cl.define_model(vocab_size, max_len, in_seq_t, inputs1_t)
	model_ref = dp_decoder_cl.define_model(vocab_size, max_len, in_seq_r, inputs1_r)

	model_tar = load_model('model_10.h5')
	model_ref = load_model('model_10.h5')

	cl_model = concatenate([model_tar, model_ref], axis=1)
	model = Model(inputs=[inputs1_t, in_seq_t, inputs1_r, in_seq_r], outputs=cl_model)
	
	adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	model.compile(loss=con_loss_fn_new, optimizer=adam)
	
	print(model.summary())	
	plot_model(model, to_file='cl_model_1.png', show_shapes=True)

	return model