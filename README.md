# Image Caption Generator
Generates image captions using an encoder-decoder model where the encoder is a CNN and the decoder is an LSTM unit. 

### README.md to be updated.

The repository is an implementation of [this paper](https://arxiv.org/abs/1710.02534) at the 31st Conference on Neural Information Processing Systems (NIPS) in 2017, titled "Contrastive Learning for Image Captioning" and authored by Bo Dai and Dahua Lin.

# Note - 
The repository is exhaustive and contains all the code files but when I wrote this project my aim was not to create an end-to-end implementation. Also, without access to a GPU I had to use some hacks to train batches manually. So naturally the repository needs some care and maintenance to make it run end-to-end. Currently it just contains all of the files I have used. I am working on making it as automated as possible whenever I can find time from other responsibilities. 

# Getting Around the Repository
Most of the functionality is in the Positive Encoder and Positive Decoder folders. The implementation uses both negative image-and-caption pairs as well as positive image-and-caption pairs to make sure that the generated captions are not only as close to the positive captions as possible but also as far from the negative ones as possible. Negative pairs are created by randomly selecting upto 5 incorrect captions for each image from the dataset.

The custom loss function, negative pair creation and some utility functions are in the Utilities folder as separate files. The generated captions are evaluated using the BLEU evaluation metric. 

# Data 
The dataset used is Microsoft's Common Objects in Context (MS COCO) which is available for download [here](http://cocodataset.org/#download).
