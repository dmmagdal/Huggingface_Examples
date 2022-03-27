# finetune_gpt2.py
# Finetune a GPT model from Huggingface with Tensorflow 2.0.
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import os
import tensorflow as tf
from tensorflow import keras
from transformers import GPT2Tokenizer
from transformers import TFGPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import pipeline
from transformers import Trainer, TFTrainer
from transformers import TrainingArguments, TFTrainingArguments


def main():
	# Build dataset from text files
	folder = "./cleaned_stories"
	files = os.listdir(folder)
	single_string = ""
	for file in files:
		with open(os.path.join(folder, file), "r", encoding="utf8") as f:
			single_string += f.read()

	# Load pre-trained GPT2 tokenizer and model.
	pretrained = "gpt2-medium"
	cache_dir = "./gpt-medium-pretrained"
	custom_tokens = ["<|title|>", "<|tags|>", "<|story|>"]
	tokenizer = GPT2Tokenizer.from_pretrained(
		pretrained, 
		cache_dir=cache_dir,
		bos_token="<|startoftext|>",
		eos_token="<|endoftext|>",
		pad_token="<|pad|>",
		additional_special_tokens=custom_tokens
	)

	config = GPT2Config()
	model = TFGPT2LMHeadModel.from_pretrained(
		pretrained, 
		cache_dir=cache_dir
	)

	# Tokenize the single string.
	string_tokenized = tokenizer.encode(single_string)

	# Create a tensorflow dataset.
	examples = []
	block_size = 100 # Number of tokens per sample
	batch_size = 12
	buffer_size = 100

	for i in range(0, len(string_tokenized) - block_size + 1, block_size):
		examples.append(string_tokenized[i:i + block_size])
	inputs, labels = [], []

	for ex in examples:
		inputs.append(ex[:-1])
		labels.append(ex[1:])

	dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
	dataset = dataset.shuffle(buffer_size).batch(
		batch_size, drop_remainder=True
	)

	print(len(list(dataset.as_numpy_iterator())))
	# exit(0)

	'''
	# Define optimizer, loss function, metrics.
	optimizer = keras.optimizers.Adam(
		learning_rate=3e-5, epsilon=1e-8, clipnorm=1.0
	)
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metric = keras.metrics.SparseCategoricalAccuracy("accuracy")

	# Compile the model.
	model.compile(
		optimizer=optimizer, metrics=[metric], 
		loss=[loss, *[None] * model.config.n_layer]
	)

	# Define hyperparameters and train.
	num_epochs = 10
	history = model.fit(dataset, epochs=num_epochs)
	'''


	#'''
	# Initialize trainer and training arguments.
	training_args = TFTrainingArguments(
		output_dir="./gpt2-finetuned",
		overwrite_output_dir=True,
		num_train_epochs=3,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		save_steps=800,
		warmup_steps=500,
		prediction_loss_only=True,
		logging_dir="./logs",
		logging_steps=10,
	)
	trainer = TFTrainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
	)

	# Train and save the model.
	trainer.train()
	trainer.save_model()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()