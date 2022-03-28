# training_clm_from_scratch.py
# Train a causal language model like GPT2 from scratch with huggingface
# transformers to generate python code.
# Source: https://huggingface.co/course/chapter7/6
# Windows/MacOS/Linux
# Python 3.7


#import torch
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline


def main():
	# This example will build a scaled down version of a code
	# generation model: focusing on one-line completions instead of
	# full functions or classes, using a subset of python code. When
	# working with data in Python, you are in frequent contact with the
	# Python data science stack, consisting of matplotlib, seaborn, 
	# pandas, and scikit-learn libraries. When using those frameworks
	# it's common to need to look up specific commands, so it would be
	# nice if we could use a model to complete these calls for us.
	# Here, we'll apply our tokenizer to a corpus of Python derived
	# from GitHub repositories. We will then use the Trainer API and
	# Huggingface Accelerate to train the model.

	# Gathering the data
	# Python code is abundantly available from code repositories such
	# as GitHub, which we can use to create a dataset by scraping for
	# every Python repository. This was th approach taken in the 
	# Transformers texbook (https://www.oreilly.com/library/view/
	# natural-language-processing/9781098103231/) to pretrain a large
	# GPT-2 model. Using a GitHub dump of about 180GB containing
	# roughly 20 million Python files called codeparrot, the authors
	# built a dataset that they then shared on the Huggingface Hub.
	# However, training on the full corpus is time and compute
	# consuming, and we only need the subset of the dataset concerned
	# with the Python data science stack. So, let's start by filtering
	# the codeparrot dataset for all files that include any of the
	# libraries in this stack. Because of hte dataset's size, we want
	# to avoid downloading it; instead, we'll use streaming feature to
	# filter it on the fly. To help us filter the code samples using
	# the libraries we mentioned earlier, we'll use the following
	# function:
	def any_keyword_in_string(string, keywords):
		for keyword in keywords:
			if keyword in string:
				return True
		return False


	# Testing it on two examples:
	filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
	example_1 = "import numpy as np"
	example_2 = "import pandas as pd"
	print(
		any_keyword_in_string(example_1, filters),
		any_keyword_in_string(example_2, filters)
	)


	# We can use this to create a function that will stream the dataset
	# and filter the elements we want:
	def filter_streaming_dataset(dataset, filters):
		filtered_dict = defaultdict(list)
		total = 0
		for sample in tqdm(iter(dataset)):
			total += 1
			if any_keyword_in_string(sample["content"], filters):
				for k, v in sample.items():
					filtered_dict[k].append(v)
		print(f"{len(filtered_dict['content']) / total:.2%} of data after filtering.")
		return Dataset.from_dict(filtered_dict)


	# Then we can simply apply this function to the streaming dataset.
	split = "train" # "valid"

	'''
	data = load_dataset(
		f"transformersbook/codeparrot-{split}", 
		split=split, 
		streaming=True
	)
	filtered_data = filter_streaming_dataset(data, filters)
	'''

	# This leaves us with about 3% of the original dataset, which is
	# still quite sizable - the resulting dataset is 6GB of 600,000
	# Python scripts.
	# Filtering the full dataset can take 2 - 3 hours depending on your
	# machine and bandwidth. If you don't want to go through this
	# lengthy process yourself, here is the filtered dataset from 
	# Huggingface hub to download.
	ds_train = load_dataset(
		"huggingface-course/codeparrot-ds-train", split="train"
	)
	ds_valid = load_dataset(
		"huggingface-course/codeparrot-ds-valid", split="train"
	)

	raw_datasets = DatasetDict(
		{
			"train": ds_train, # .shufflt().select(range(50000))
			"valid": ds_valid, # .shufflt().select(range(500))
		}
	)
	print(raw_datasets)

	# Pretraining the language model will take a while. It is suggested
	# that you first run the training loop on a sample of the data by
	# uncommenting the two partial lines above, and make sure that the
	# training successfully completes and the models are stored.
	# Nothing is more frustrating than a training run failing at the
	# last step because you forgot to create a folder or because
	# there's a typo at the end of the training loop.

	# Looking at an example from the dataset. We'll just show the first
	# 200 characters of each field.
	for key in raw_datasets["train"][0]:
		print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

	# We can see that the content field contains te code that we want
	# our model to train on. Now that we have a dataset, we need to
	# prepare the texts so they're in a format suitable for
	# pretraining. 

	# Prepare the dataset
	# The first step will be to tokenize the dataset, so we can use it
	# for training. Since our goal is to mainly autocomplete short
	# function calls, we can keep the context size relatively small.
	# This has the benefit that we can train the model much faster and
	# it requires significantly less memory. If it is important for
	# your application to have more context (for example, if you want
	# the model to write unit tests based on a file with the function
	# defenition), make sure you increase that number, but also keep in
	# mind that this comes with a greater GPU memory footprint. For
	# now, let's fix the context size at 128 tokens, as opposed to the
	# 1024 or 2048 used in GPT-2 or GPT-3 respectively.
	# Most documents contain many more than 128 tokens, so simply
	# truncating the inputs to the maximum length would eliminate a
	# large fraction of our dataset. Instead, we'll use the 
	# return_overflowing_tokens option to tokenize the whole input and
	# split it into several chunks. We'll also use the return_length
	# option to return the length of each created chunk automatically.
	# Often the last chunk will be smaller than the context size, and
	# we'll get rid of these pieces to avoid padding issues; we don't
	# really need them as we have plenty of data anyway.
	context_length = 128
	tokenizer = AutoTokenizer.from_pretrained(
		"huggingface-course/code-search-net-tokenizer"
	)

	outputs = tokenizer(
		raw_datasets["train"][:2]["content"],
		truncation=True,
		max_length=context_length,
		return_overflowing_tokens=True,
		return_length=True,
	)

	print(f"Input IDs length: {len(outputs['input_ids'])}")
	print(f"Input chunk lengths: {len(outputs['length'])}")
	print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")


	# We can see that we get 34 segments in totalt from those two
	# examples. Looking at the chunk lengths, we can see that the
	# chunks at the ends of both documents have less than 128 tokens
	# (117 and 41, respectively). These represent just a small fraction
	# of the total chunks that we have, so we can safely throw them
	# away. With the overflow_to_sample_mapping field, we can also
	# reconstruct which chunks belonged to which input samples.
	# With this operation we're using a handy feature of the
	# Dataset.map() function in huggingface Datasets, which is that it
	# does not require one-to-one maps, we can create batches with more
	# or fewer elements than the input batch. This si useful when doing
	# operations like data augmentation or data filtering that change
	# the number of elements. In our case, when tokenizing each
	# element into chunks of the specified context size, we create many
	# samples form each document. We just need to make sure to delete
	# the existing columns, since they have a conflicting size. If we
	# wanted to keep them, we could repeat them appropriately and
	# return them within the Dataset.map() call:
	def tokenize(element):
		outputs = tokenizer(
			element["content"],
			truncation=True, 
			max_length=context_length,
			return_overflowing_tokens=True,
			return_length=True,
		)
		input_batch = []
		for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
			if length == context_length:
				input_batch.append(input_ids)
		return {"input_ids": input_batch}


	tokenized_datasets = raw_datasets.map(
		tokenize, 
		batched=True, 
		remove_columns=raw_datasets["train"].column_names
	)
	print(tokenized_datasets)

	# We now have 16.7 million examples with 128 tokens each, which
	# corresponds to about 2.1 billion tokens in total. For reference,
	# OpenAI's GPT-3 and Codex models are trained on 300 and 100
	# billion tokens, respectively, where the Codex models are 
	# initialized from the GPT-3 checkpoints. Our goal in this section
	# is not to create a scaled-down version providing a quick
	# autocomplete function for data scientists.
	# Now that we have the dataset ready, let's set up the model.

	# Initializing a new model
	# Our first step is to freshly initialize a GPT-2 model. We'll use
	# the same configuration for our model as for the small GPT-2
	# model, so we load the pretrained configuration, make sure that
	# the tokenizer size matches the model vocabulary size and pass the
	# bos and eos (beginning and end of sequence) token IDs:
	config = AutoConfig.from_pretrained(
		"gpt2",
		vocab_size=len(tokenizer),
		n_ctx=context_length,
		bos_token_id=tokenizer.bos_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)

	# With that configuration, we can load a new model. Note that this
	# is the first time we don't use the from_pretrained() function,
	# since we're actually initializing a model ourself.
	model = GPT2LMHeadModel(config)
	model_size = sum(t.numel() for t in model.parameters())
	print(f"GPT-2 size: {model_size / 1000**2:.1f}M parameters")

	# Our model has 124M parameters that we'll have to tune. Before we
	# can start training, we need to set up a data collator that will
	# take care of creating the batches. We can use the 
	# DataCollatorForLanguageModeling collator, which is designed 
	# specifically for language modeling (as the name subtly suggests).
	# besides stacking and padding batches, it also takes care of
	# creating the language model labels - in causal language modeling
	# the inputs server as labels too (just shifted by one element),
	# and this data collator creates them on the fly during training so
	# we don't duplicate the input_ids.
	# Note that DataCollatorForLanguageModeling supports both masked
	# language modeling (MLM) and causal language modeling (CLM). By
	# default it prepares data for MLM, but we can switch to CLM by
	# setting the argument mlm=False.
	tokenizer.pad_token = tokenizer.eos_token
	data_collator = DataCollatorForLanguageModeling(
		tokenizer, 
		mlm=False
	)

	# Here is an example:
	out = data_collator(
		[tokenized_datasets["train"][i] for i in range(5)]
	)
	for key in out:
		print(f"{key} shape: {out[key].shape}")

	# We can see that the examples have been stacked and all the
	# tensors have the same shape. Note: Shifting the inputs and labels
	# to align them happens inside the model, so the data collator just
	# copies the inputs and creates the labels.
	# Now we have everything in place to actually train our model = 
	# that wasn't so much work after all! Before we start training we
	# should log in to Huggingface. If working in a notebook, you can
	# do so with the following utility function:
	# from huggingface_hub import notebook_login
	# notebook_login()
	# This will display a widget where you can enter your huggingface
	# login credentials.
	# If not using a notebook, type the following into the terminal:
	# huggingface-cli login
	# All that's left to do is configure the training arguments and 
	# fire up the Trainer. We'll use a cosine learning rate schedule
	# with some warmup and an effective batch size of 256
	# (per_device_train_batch_size * gradient_accumulation_steps).
	# Gradient accumulation is used when a single batch does not fit
	# into memory, and incrementally builds up the gradient through
	# several forward/backward passes. We'll see this in action when
	# we create the training loop with Huggingface Accelerate.
	args = TrainingArguments(
		output_dir="codeparrot-ds",
		per_device_train_batch_size=32,
		per_device_eval_batch_size=32,
		evaluation_strategy="steps",
		eval_steps=5_000,
		logging_steps=5_000,
		gradient_accumulation_steps=8,
		num_train_epochs=1,
		weight_decay=0.1,
		lr_scheduler_type="cosine",
		learning_rate=5e-4,
		save_steps=5_000,
		fp16=True,
		push_to_hub=False,
	)
	trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=args,
		data_collator=data_collator,
		train_dataset=tokenized_datasets['train'],
		eval_dataset=tokenized_datasets['valid'],
	)

	# Now we can just start the Trainer and wait for training to
	# finish. Depending on whether you run it on the full or a subset
	# of the training set this will take 20 or 2 hours, respectively,
	# so grab a few coffees and a good book to read!
	trainer.train()

	# After training completes, we can push the model and tokenizer to
	# the hub:
	# trainer.push_to_hub()

	# Code generation with a pipeline
	# Now is the moment of truth: let's see how well the trained model
	# actually works. We can see in the logs that the loss went down
	# steadily, but to put the model to the test, let's take a look at
	# how well it works on some prompts. To do that, we'll wrap the
	# model in a text generation pipeline, and we'll put it on the GPU
	# for fast generations if there is one available:
	#device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
	pipe = pipeline(
		"text-generation", 
		model="huggingface-course/codeparrot-ds",
		#device=device,
	)

	# Let's start with the simple task of creating a scatter plot.
	txt = '''
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
'''
	print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

	# Looking at these few examples, it seems that the model has
	# learned some of the syntax of the Python data science stack (of
	# course, we would need to evaluate it more thoroughly before
	# deploying the model in the real world). Sometimes it requires
	# more customization of the model training to achieve the necessary
	# performance for a given use case, howerver. For examples, what if
	# we would like to dynamically update the batch size or have a
	# conditional training loop that skips bad examples on the fly? One
	# option would be to subclass the Trainer and add the necessary
	# changes, but sometimes it's simpler to write the training loop
	# from scratch. That's where Huggingface Accelerate comes in.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()