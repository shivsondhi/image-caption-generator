import ep_generate
import vp_tokens
import vp_fit_and_train
from keras.models import load_model


if __name__ == "__main__":
	val = vp_fit_and_train.load_set("validation_id_captions.txt")
	print('Size of validation dataset: {}'.format(len(val)))

	val_descs = vp_fit_and_train.load_clean_desc('validation_id_captions.txt', val, 'validation_descriptions.json')
	print("Validation descriptions: {}".format(val_descs))
	
	#photo features
	val_features = vp_fit_and_train.load_photo_features('validation_features.pkl', val)
	print('Validation Photos: {}'.format(len(val_features)))

	tokenizer = vp_tokens.create_tokenizer("validation_descriptions.json")

	max_len = vp_tokens.max_length("validation_descriptions.json")
	print("Maximum caption length: {}".format(max_len))

	#num = epoch with best loss
	filename = 'saved_models\\model_{}.h5'.format(str(7))
	model = load_model(filename)
	
	#evaluate model
	ep_generate.evaluate_model(model, "validation_descriptions.json", val_features, tokenizer, 49)
