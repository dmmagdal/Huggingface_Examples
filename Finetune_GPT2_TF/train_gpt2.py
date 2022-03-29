# train_gpt2.py
# Train GPT-2 with Huggingface transformers and Tensorflow 2.
# Source https://towardsdatascience.com/train-gpt-2-in-your-own-language-fc6ad4d60171
# Source (Kaggle Arxiv Dataset): https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
# Source (Huggingface Reference): https://huggingface.co/blog/how-to-generate
# Source (Personal 10X Github): https://github.com/diegomagdaleno10xgenomics/python/blob/main/tf_gpt2/train_gpt2.py
# Tensorflow 2.7.0
# Windows/MacOS/Linux
# Python 3.7


import os
import json
import tensorflow as tf
from tensorflow import keras
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer


def main():
	# Initialize model.
	pretrained = "gpt2-medium"
	cache_dir = "./gpt2-pretrained"
	tokenizer = GPT2Tokenizer.from_pretrained(
		pretrained, cache_dir=cache_dir
	)

	config = GPT2Config(
		vocab_size=tokenizer.vocab_size,
		bos_token_id=tokenizer.bos_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)

	model = TFGPT2LMHeadModel(config)

	# Create a single string from all documents.
	single_string = ""

	# Use for having multiple documents as part of the dataset.
	folder = "./stories"
	paths = os.listdir(folder)
	for file in paths:
		with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
			#x = f.read()
			contents = f.readlines()

			# Prepend special character to title.
			contents[0] = contents[0].rstrip("\n")

			# indices 1, 2, 3, 4 are the author, description, story
			# link and author page link respectively. We do not want
			# those for the dataset. This is format is constant from
			# the way literotica_crawler saves stories.

			# Prepend special character to tags.
			contents[5] = contents[5].rstrip("\n")

			# Prepend special character to main story.
			contents[6] = " ".join(
				[content.rstrip("\n") for content in contents[6:]]
			)
			contents = [contents[0]] + [contents[5]] + contents[6:]
			x = " ".join(contents)
		single_string += x + tokenizer.eos_token

	string_tokenized = tokenizer.encode(single_string)

	# Create TF Dataset. The following hyperparameters (block_size 128,
	# batch_size 12, buffer_size 100) took almost all 8GB of VRAM on my
	# 2060 SUPER.
	examples = []
	block_size = 128
	batch_size = 12
	buffer_size = 100

	for i in range(0, len(string_tokenized) - block_size + 1, block_size):
		examples.append(string_tokenized[i:i + block_size])
	inputs, labels = [], []

	for ex in examples:
		inputs.append(ex[:-1])
		labels.append(ex[1:])

	dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
	dataset = dataset.shuffle(buffer_size)\
		.batch(batch_size, drop_remainder=True)

	# Initialize optimizer, loss, and metrics. The compile the mdoel.
	optimizer = keras.optimizers.Adam(
		learning_rate=3e-5, epsilon=1e-8, clipnorm=1.0
	)
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	metric = keras.metrics.SparseCategoricalAccuracy("accuracy")

	model.compile(
		optimizer=optimizer, 
		loss=[loss, *[None] * model.config.n_layer],
		metrics=[metric] 
	)

	# Train the model.
	num_epoch = 10
	history = model.fit(dataset, epochs=num_epoch)

	# Save the model.
	model.save_pretrained("./gpt2-stories")
	tokenizer.save_pretrained("./gpt2-stories")

	# Have the model run inference.
	text = "Veronica stepped out of the pool, making her way over to"
	input_ids = tokenizer.encode(text, return_tensors="tf")
	beam_output = model.generate(
		input_ids,
		max_length=100,
		num_beams=5,
		temperature=0.7,
		no_repeat_ngram_size=2,
		num_return_sequences=5
	)

	print("PROMPT >" + text)
	print("-" * 72)
	for i in range(len(beam_output)):
		print(f"OUTPUT {i + 1}: {tokenizer.decode(beam_output[i])}")

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main() 