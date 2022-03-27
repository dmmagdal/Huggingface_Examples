# finetune_gpt2_alt_tf.py
# Finetune a GPT model from Huggingface with Tensorflow 2.0.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import tensorflow as tf
from tensorflow import keras
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, AutoTokenizer
from transformers import TFGPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import pipeline
from transformers import TFTrainer, TFTrainingArguments


def main():
	# Build dataset from text files using Huggingface's Datasets
	# library.
	folder = "./cleaned_stories"
	files = os.listdir(folder)
	stories = []
	for file in files:
		with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
			stories.append(f.read())

	# Define the train/eval split.
	split = 0.7
	train_split = int(len(stories) * split)
	training_data = stories[:train_split]
	val_split = len(stories) - train_split
	validation_data = stories[-val_split:]

	# Set up a context length as well as load a pretrained (GPT-2)
	# tokenizer.
	context_length = 256
	pretrained = "gpt2-medium"
	cache_dir = "./gpt-medium-pretrained"
	custom_tokens = ["<|title|>", "<|tags|>", "<|story|>"]
	tokenizer = AutoTokenizer.from_pretrained(
		pretrained, 
		cache_dir=cache_dir,
		bos_token="<|startoftext|>",
		eos_token="<|endoftext|>",
		pad_token="<|pad|>",
		additional_special_tokens=custom_tokens
	)

	# Tokenize the (training) dataset (on the first two samples) as an
	# example. Note that using the AutoTokenizer instead of the
	# GPT2Tokenizer will have 'overflow_to_sample_mapping' as a key in
	# the outputs.
	outputs = tokenizer(
		training_data[:2],
		truncation=True,
		max_length=context_length,
		return_overflowing_tokens=True,
		return_length=True,
	)
	print(f"Input IDs length: {len(outputs['input_ids'])}")
	print(f"Input chunk lengths: {outputs['length']}")
	print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")


	# Map the tokenize() function to all elements in the raw datasets
	# (train and valid). This will tokenize all the texts in the
	# 'contents' column and remove all other data, leaving only the
	# tokenized 'input_ids' of the texts.
	def tokenize(element):
		outputs = tokenizer(
			element,
			truncation=True,
			max_length=context_length,
			return_overflowing_tokens=True,
			return_length=True,
		)
		input_batch = []
		for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
			if length == context_length:
				input_batch.append(input_ids)
		# return {"input_ids": input_batch}

		# Map input_ids and output_ids.
		input_ids = input_batch[:-1]
		label_ids = input_batch[1:]
		return {"input_ids": input_ids, "label_ids": label_ids}


	# tokenized_training_data = tokenize(training_data)
	# tokenized_validation_data = tokenize(validation_data)
	train_dataset = tf.data.Dataset.from_tensor_slices(
		tokenize(training_data)
	)
	valid_dataset = tf.data.Dataset.from_tensor_slices(
		tokenize(validation_data)
	)
	print(list(train_dataset.as_numpy_iterator())[0])
	print(train_dataset)

	# Load pretrained (GPT-2) model.
	config = GPT2Config.from_pretrained(
		pretrained,
		cache_dir=cache_dir,
		vocab_size=len(tokenizer),
		n_ctx=context_length,
		bos_token_id=tokenizer.bos_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)
	model = TFGPT2LMHeadModel(config)
	model.resize_token_embeddings(len(tokenizer))

	'''
	# Initialize data collator to handle batch creation (as well as
	# creating the language model labels). Given that the current
	# data collator can also do masked language modeling (mlm) as well
	# as causal language modeling (clm). 
	tokenizer.pad_token = tokenizer.eos_token
	data_collator = DataCollatorForLanguageModeling(
		tokenizer, mlm=False, return_tensors="pt"#return_tensors="tf"
	)

	# An example of the output from the data collator.
	out = data_collator(
		[tokenized_datasets["train"][i] for i in range(5)]
	)
	for key in out:
		print(f"{key} shape: {out[key].shape}")
	'''

	# Initialize the training arguments and the trainer.
	# args = TFTrainingArguments(
	# 	output_dir="./gpt2-medium-finetuned",
	# 	per_device_train_batch_size=8,
	# 	per_device_eval_batch_size=8,
	# 	evaluation_strategy="steps",
	# 	eval_steps=500,
	# 	logging_steps=500,
	# 	gradient_accumulation_steps=8,
	# 	num_train_epochs=3,
	# 	weight_decay=0.1,
	# 	warmup_steps=100,
	# 	lr_scheduler_type="cosine",
	# 	learning_rate=5e-4,
	# 	save_steps=500,
	# 	#fp16=True, # Only use with CUDA devices.
	# )

	optimizer = keras.optimizers.Adam(learning_rate=3e-5)
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metric = keras.metrics.SparseCategoricalAccuracy("accuracy")
	# model.build(input_shape=(None, context_length))
	outs = model(model.dummy_inputs)
	model.summary()
	print(model.dummy_inputs)
	exit()
	model.compile(
		optimizer=optimizer, loss=loss, metrics=[metric]
	)

	# Train the model.
	history = model.fit(
		train_dataset,
		epochs=3,
		validation_data=valid_dataset,
	)
	exit(0)



	# Load the model to pipeline.
	pipe = pipeline(
		"text-generation", model="./gpt2-medium-finetuned-final"
	)

	# Sample some output from the pipeline.
	txt1 = "Veronica stepped out of the pool, making her way over to"
	txt2 = "Jesse grabbed the railing, hoping that Emily would"
	txt3 = "\"What are you doing?\" Amir heard a voice shout behind him"


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()