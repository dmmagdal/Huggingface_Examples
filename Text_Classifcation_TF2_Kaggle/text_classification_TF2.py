# text_classification_TF2.py
# Perform multi-label text classifcation with the Kaggle "Toxic Comment
# Classification" challenge using huggingface transformers and
# tensorflow 2.
# Source: https://towardsdatascience.com/working-with-hugging-face-
# transformers-and-tf-2-0-89bf35e3555a
# Source (data): https://www.kaggle.com/c/jigsaw-toxic-comment-
# classification-challenge/data?select=test.csv.zip
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import DistilBertTokenizer, RobertaTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertConfig
from transformers import TFDistilBertModel
from transformers import AutoTokenizer, pipeline


def main():
	# The following is a general pipeline for any transformer model:
	# tokenizer definition -> tokenization of documents -> model
	# definition -> model training -> inference.

	# Tokenizer definition
	# Every transformer based model has a unique tokenization
	# technique and unique use of special tokens. The transformer
	# library takes care of this for us. It supports tokenization for
	# every model which is associated with it.
	distil_bert = "distilbert-base-uncased" # Pick any desired pre-trained model
	roberta = "roberta-base"

	# Defining DistilBERT tokenizer.
	tokenizer = DistilBertTokenizer.from_pretrained(
		distil_bert, do_lower_case=True, add_special_tokens=True,
		max_length=128, pad_to_max_length=True
	)

	# Defining RoBERTa tokenizer.
	tokenizer = RobertaTokenizer.from_pretrained(
		roberta, do_lower_case=True, add_special_tokens=True,
		max_length=128, pad_to_max_length=True
	)

	# Here:
	# -> add_special_tokens: is used to add special characters like
	#	<cls>, <sep>, <unk>, etc. W.R.T pretrained model in use. It
	#	should always be kept True.
	# -> max_length: max length of any sentence to tokenize, its a
	#	hyperparameter (originally BERT has 512 max length).
	# -> pad_to_max_length: perform padding operation.


	# Tokenization of documents
	# Next step is to perform tokenization on documents. It can be
	# performed either by encode() or encode_plus() method.
	def tokenize(sentences, tokenizer):
		input_ids, input_masks, input_segments = [], [], []
		for sentence in tqdm(sentences):
			inputs = tokenizer.encode_plus(
				sentence, add_special_tokens=True, max_length=128,
				pad_to_max_length=True, return_attention_mask=True,
				return_token_type_ids=True
			)
			input_ids.append(inputs["input_ids"])
			input_masks.append(inputs["attention_mask"])
			input_segments.append(inputs["token_type_ids"])

		return (
			np.asarray(input_ids, dtype="int32"),
			np.asarray(input_masks, dtype="int32"),
			np.asarray(input_segments, dtype="int32"),
		)


	# Any transformer model generally needs three inputs:
	# -> input ids: word id associated with their vocabulary.
	# -> attention_mask: which id must be paid attention to; 1 = pay
	#	attention. In simple terms, it tells themodel which are 
	#	original words and which are padded words or special tokens.
	# -> token type id: it's associated with models consuming multiple
	#	sentences like question-answer models. It tells model about the
	#	sequence of the sentences.
	# Though it is not compulsory to provide all three ids, only input
	# ids will also do. But attention mask help model to focus on only
	# valid words. So at least for classification task both of these
	# should be provided.

	# Training and fine-tuning
	# There are three possible ways to train the model:
	# 1) Use pre-trained model directly as a classifer.
	# 2) Transformer model to extract embedding and use it as input to
	#	another classifier.
	# 3) Fine-tuning a pre-trained transformer model on custom config
	#	and dataset.

	# Training Method 1: Use pre-trained model directly as classifier
	# This is the simplest but also with the least application.
	# Huggingface's transformers library provides some models with
	# sequence classification ability. These models have two heads, one
	# is a pre-trained model architecture as the base and a classifier
	# as the top head.
	# Tokenizer definition -> tokenization of documents -> model
	# definition.
	distil_bert = "distilbert-base-uncased"

	config = DistilBertConfig(num_labels=6)
	config.output_hidden_states = False
	transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
		distil_bert, config=config
	)

	input_ids = layers.Input(
		shape=(128,), name="input_token", dtype="int32"
	)
	input_masks_ids = layers.Input(
		shape=(128,), name="masked_token", dtype="int32"
	)
	X = transformer_model(input_ids, input_masks_ids)
	model = keras.Model(inputs=[input_ids, input_masks_ids], outputs=X)
	print(model.summary())

	# Note: models which are SequenceClassifcation are only applicable
	# here. Defining the proper config is crucial here. As you can see,
	# defining the config "num_labels" is the number of classes to use
	# when the model is a classication model. It also supports a
	# variety of configs so go ahead & see their docs.
	# Some key things to note here:
	# -> Here only weights of the pre-trained model can be updated, but
	#	updating them is not a good idea as it will defeat the purpose
	#	of transfer learning. So, actually there is nothing here to
	#	update.
	# -> It is also the least customizable.
	# -> A hack you can try is using num_labels with much higher value
	#	and finally adding a dense layer at the end which can be 
	#	trained.

	# Hack.
	config = DistilBertConfig(num_labels=64)
	config.output_hidden_states = False
	transformer_model = TFDistilBertForSequenceClassification.from_pretrained(
		distil_bert, config=config
	)

	input_ids = layers.Input(
		shape=(128,), name="input_token", dtype="int32"
	)
	input_masks_ids = layers.Input(
		shape=(128,), name="masked_token", dtype="int32"
	)
	X = transformer_model(input_ids, input_masks_ids)[0]
	X = layers.Dropout(0.2)(X)
	X = layers.Dense(6, activation="softmax")(X)
	model = keras.Model(inputs=[input_ids, input_masks_ids], outputs=X)
	for layer in model.layers[:2]:
		layer.trainable = False
	print(model.summary())

	# Training Method 2: Transformer model to extract embedding and use
	# it as input to another classifier
	# This approach needs two level or two separate models. Use any
	# transformer model to extract word embedding and then use this
	# word embedding as input to any classifier (e.g. logistic
	# classifier, random forest, neural nets, etc). Read this blog post
	# (http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-
	# first-time/) by Jay Alammar which discusses this approach with
	# great detail and clarity.
	distil_bert = "distilbert-base-uncased"

	config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
	config.output_hidden_states = False
	transformer_model = TFDistilBertModel.from_pretrained(
		distil_bert, config=config
	)

	input_ids_in = layers.Input(
		shape=(128,), name="input_token", dtype="int32"
	)
	input_masks_in = layers.Input(
		shape=(128,), name="masked_token", dtype="int32"
	)

	embedding_layer = transformer_model(
		input_ids_in, attention_mask=input_masks_in
	)[0]
	cls_token = embedding_layer[:, 0, :]
	X = layers.BatchNormalization()(cls_token)
	X = layers.Dense(192, activation="relu")(X)
	X = layers.Dropout(0.2)(X)
	X = layers.Dense(6, activation="softmax")(X)
	model = keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)
	for layer in model.layers[:3]:
		layer.trainable = False
	print(model.summary())

	# Referring to the cls_token line, we are only interested in <cls>
	# or classification token of the model which can be extracted using
	# the slice operation. Now we have 2D data and build the network as
	# one desired.
	# This approach works generall better every time compared to the
	# first approach mentioned above. But it also has some drawbacks:
	# -> It is not suitable for production, as you must be using
	#	transformer model as just a feature extractor and so you have
	#	to now maintain two models, as your classifier head is
	#	different (like XGBoost or CatBoast).
	# -> While converting 3D data to 2D we may miss on valuable info.
	# The transformers library provides a great utility if you want to
	# just extract work embeddings.

	# Hack.
	model = TFDistilBertModel.from_pretrained(distil_bert)

	tokenizer = AutoTokenizer.from_pretrained(distil_bert)

	pipe = pipeline(
		"feature-extraction", model=model, tokenizer=tokenizer
	)

	features = pipe(
		"any text data or list of text data", pad_to_max_length=True
	)
	features = np.squeeze(features)
	#features = features[:, 0, :]
	features = features[:, 0]

	# Training Method 3: Fine-tuning a pre-trained transformer model
	# Here we are making use of the full potential of any transformer
	# model. We'll be using the weights of pre-trained transformer
	# models and then fine-tune our data (ie transfer learning).
	distil_bert = "distilbert-base-uncased"

	config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
	config.output_hidden_states = False
	transformer_model = TFDistilBertModel.from_pretrained(
		distil_bert, config=config
	)

	input_ids_in = layers.Input(
		shape=(128,), name="input_token", dtype="int32"
	)
	input_masks_in = layers.Input(
		shape=(128,), name="masked_token", dtype="int32"
	)

	embedding_layer = transformer_model(
		input_ids_in, attention_mask=input_masks_in
	)[0]
	X = layers.Bidirectional(
		layers.LSTM(
			50, return_sequences=True, dropout=0.1,
			recurrent_dropout=0.1
		)
	)(embedding_layer)
	X = layers.GlobalMaxPool1D()(X)
	X = layers.Dense(50, activation="relu")(X)
	X = layers.Dropout(0.2)(X)
	X = layers.Dense(6, activation="sigmoid")(X)
	model = keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)
	for layer in model.layers[:3]:
		layer.trainable = False
	print(model.summary())

	# Every approach has two things in common:
	# 1) config.output_hidden_states = False; as we are training and
	#	not interested in output state.
	# 2) X = transformer_model(..)[0]; this isinline in 
	#	config.output_hidden_states as we want only the top head.
	# Config is a dictionary. 
	# Choose a base model carefully as TF 2.0 support is new, so there
	# might be bugs.

	# Inference
	# As the model is based on tf.keras models API, we can use Keras'
	# same commonly used method of model.predict(). We can even use
	# the transformer library's pipeline utility (refer to the second
	# model training example). This utility is quite effective as it
	# unifies tokenization and prediction under one common simple API.

	# End notes
	# Huggingface has really made it quite easy to make use of any of
	# their models now with tensorflow keras. They have also made it
	# quite easy to use their model in the cross library (from pytorch
	# to tensorflow or vice versa).

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()