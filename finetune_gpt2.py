# finetune_gpt2.py
# Walk through Medium example of fine-tuning GPT2 with Huggingface
# transformers.
# Source: https://medium.com/swlh/fine-tuning-gpt-2-for-magic-the-
# gathering-flavour-text-generation-3bafd0f9bb93
# Source (Colab): https://colab.research.google.com/drive/
# 16UTbQOhspQOF3XlxDFyI28S-0nAkTzk_
# Source (Colab): https://drive.google.com/file/d/
# 16UTbQOhspQOF3XlxDFyI28S-0nAkTzk_/view
# Windows/MacOS/Linux
# Python 3.7


import os
import time
import datetime
import random
import requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from itertools import compress
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup


def main():
	# Download the training dataset (Magic the Gathering cards).
	response = requests.get(
		"https://c2.scryfall.com/file/scryfall-bulk/all-cards/all-cards-20200831091816.json"
	)
	data = response.json()

	# Start parsing by removing any cards with no flavor text.
	contains_x = []
	for i in data:
		contains_x.append("flavor_text" in i.keys())

	data_filtered = list(compress(data, contains_x))

	# Remove any cards in a language other than English.
	contains_y = []
	for i in data_filtered:
		contains_y.append("lang" in i.keys() and "en" == i["lang"])

	data_filtered = list(compress(data_filtered, contains_y))

	# Create a list to iterate through.
	cardValues = []
	for i in data_filtered:
		cardValues.append(i["flavor_text"])

	# Convert this to a dataframe to visualize a few rows nicely mostly
	# just a sanity check.
	df = pd.DataFrame(cardValues, columns=["data"])
	cards = df.data.copy()

	# Initialize a GPT-2 tokenizer.
	tokenizer = GPT2Tokenizer.from_pretrained(
		"gpt2",
		bos_token="<|startoftext|>",
		eos_token="<|endoftext|>",
		pad_token="<|pad|>"
	)

	max_flavor = max([len(tokenizer.encode(card)) for card in cards])
	print(f"The longest flavor text is {max_flavor} tokens long.")

	# Batch size.
	bs = 32


	class MTGDataset(Dataset):
		def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=max_flavor):
			self.tokenizer = tokenizer
			self.input_ids = []
			self.attn_masks = []

			# Iterate through ecah entry in the flavor text corpus.
			# Prepend the start of text token, append the end of text
			# token, and pad to the maximum length with the pad token.
			for txt in txt_list:
				encodings_dict = tokenizer(
					"<|startoftext|>" + txt + "<|endoftext|>",
					truncation=True,
					max_length=max_length,
					padding="max_length",
				)

				# Each iteration appends either the encoder tensor to a
				# list, or the attention mask for that encoding to a
				# list. The attention mask is a binary list of 1's or
				# 0's which determine whether the language model should
				# take that token into consideration or not.
				self.input_ids.append(
					torch.tensor(encodings_dict["input_ids"])
				)
				self.attn_masks.append(
					torch.tensor(encodings_dict["attention_mask"])
				)


		def __len__(self):
			return len(self.input_ids)


		def __getitem__(self, idx):
			return self.input_ids[idx], self.attn_masks[idx]


	# Initialize the dataset.
	dataset = MTGDataset(cards, tokenizer, max_length=max_flavor)

	# Split into training and validation sets.
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size

	train_dataset, val_dataset = random_split(
		dataset, [train_size, val_size]
	)
	print(f"There are {train_size} samples for training, and {val_size} samples for validation.")

	# Create DataLoaders.
	train_dataloader = DataLoader(
		train_dataset,
		sampler=RandomSampler(train_dataset),
		batch_size=bs
	)
	validation_dataloader = DataLoader(
		val_dataset,
		sampler=SequentialSampler(val_dataset),
		batch_size=bs
	)

	# Finetune GPT2 model.
	config = GPT2Config.from_pretrained(
		"gpt2", output_hidden_states=False, cache_dir="./gpt2-config"
	)

	model = GPT2LMHeadModel.from_pretrained(
		"gpt2", config=config, cache_dir="./gpt2-model"
	)
	model.resize_token_embeddings(len(tokenizer))

	# Tell pytorch to run this model on GPU (if available).
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)

	# Seed.
	seed_val = 42
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	# Training variables.
	epochs = 4
	warmup_steps = 1e2
	sample_every = 100

	# Initialize optimizer.
	optimizer = AdamW(model.parameters(), lr=5e-4, eps=1e-8)

	# Learning rate scheduler.
	total_steps = len(train_dataloader) * epochs
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=warmup_steps, 
		num_training_steps=total_steps
	)


	def format_time(elapsed):
		return str(datetime.timedelta(seconds=int(round(elapsed))))


	total_t0 = time.time()

	training_stats = []
	model = model.to(device)

	# Training loop.
	for epoch_i in range(0, epochs):
		print(f"Beginning epoch {epoch_i + 1} of {epochs}")
		t0 = time.time()
		total_train_loss = 0
		model.train()

		for step, batch in enumerate(train_dataloader):
			b_input_ids = batch[0].to(device)
			b_labels = batch[0].to(device)
			b_masks = batch[1].to(device)

			model.zero_grad()

			outputs = model(
				b_input_ids, labels=b_labels, attention_mask=b_masks,
				token_type_ids=None
			)
			loss = outputs[0]

			batch_loss = loss.item()
			total_train_loss += batch_loss

			# Get sample every 100 batches.
			if step % sample_every == 0 and not step == 0:
				elapsed = format_time(time.time() - t0)
				print(f"Batch {step} of {len(train_dataloader)}. Loss:{batch_loss}. Time:{elapsed}")

				model.eval()

				sample_outputs = model.generate(
					bos_token_id=random.randint(1, 30000),
					do_sample=True,
					top_k=50,
					max_length=200,
					top_p=0.95,
					num_return_sequences=1
				)
				for i, sample_output in enumerate(sample_outputs):
					print(f"Example output: {tokenizer.decode(sample_output, skip_special_tokens=True)}")

				model.train()

			loss.backward()
			optimizer.step()
			scheduler.step()

		# Calculate the average loss over all of the batches.
		avg_train_loss = total_train_loss / len(train_dataloader)

		# Measure how long this epoch took.
		training_time = format_time(time.time() - t0)
		print(f"Average Training Loss: {avg_train_loss}. Epoch time: {training_time}")

		t0 = time.time()
		model.eval()
		total_eval_loss = 0
		nb_eval_steps = 0

		# Evaluate data for one epoch.
		for batch in validation_dataloader:
			b_input_ids = batch[0].to(device)
			b_labels = batch[0].to(device)
			b_masks = batch[1].to(device)

			with torch.no_grad():
				outputs = model(
					b_input_ids, attention_mask=b_masks, 
					labels=b_labels
				)
				loss = outputs[0]

			batch_loss = loss.item()
			total_eval_loss += batch_loss

		avg_val_loss = total_eval_loss / len(validation_dataloader)
		validation_time = format_time(time.time() - t0)
		print(f"Validation loss: {avg_val_loss}. Validation Time: {validation_time}")

		# Record all statistics from this epoch.
		training_stats.append(
			{
				"epoch": epoch_i + 1,
				"Training Loss": avg_train_loss,
				"Valid Loss": avg_valid_loss,
				"Training Time": training_time,
				"Validation Time": validation_time,
			}
		)

	print(f"Total training took {format_time(time.time() - total_t0)}")

	# Saving and loading fine-tuned model
	output_dir = "./content/drive/MTG"

	# Save trained model, configuration, and tokenizer.
	model_to_save = model.module if hasattr(model, 'module') else model
	model_to_save.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)
	torch.save(args, os.path.join(output_dir, "training_args.bin"))

	# Generate text
	model.eval()
	prompt = "<|startoftext|>"

	generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
	generated = generated.to(device)

	sample_outputs = model.generate(
		generated,
		do_sample=True,
		top_k=50,
		max_length=300,
		top_p=0.95,
		num_return_sequences=3
	)

	for i, sample_output in enumerate(sample_outputs):
		print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

	model = GPT2LMHeadModel.from_pretrained(output_dir)
	tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
	model.to(device)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()