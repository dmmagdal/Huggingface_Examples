# finetune_gpt2_alt.py
# Finetune a GPT model from Huggingface.
# Source (HF CLM training): https://huggingface.co/course/chapter7/6
# Source (HF GPT2): https://huggingface.co/docs/transformers/model_
# doc/gpt2
# Source (Datasets): https://huggingface.co/docs/datasets/package_
# reference/main_classes
# Source (Tokenizer): https://huggingface.co/docs/transformers/main_
# classes/tokenizer
# Source (Trainer): https://huggingface.co/docs/transformers/main_
# classes/trainer
# Source (Data Collator): https://huggingface.co/docs/transformers/
# main_classes/data_collator
# Source (Docker): https://hub.docker.com/u/huggingface
# Source (Relevant HR Forum): https://discuss.huggingface.co/t/fine-
# tuning-gpt2-for-text-generation-with-tensorflow/15348
# Windows/MacOS/Linux
# Source (HF Custom Datasets): https://huggingface.co/transformers/
# v3.2.0/custom_datasets.html.
# Source (Colab GPT-2 Training): https://colab.research.google.com/
# github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_
# English_GPT_2_Model_with_Huggingface.ipynb
# Python 3.7


import os
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Config
from transformers import GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import pipeline


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
	train_dataset = Dataset.from_dict({"contents": training_data})
	valid_dataset = Dataset.from_dict({"contents": validation_data})
	raw_datasets = DatasetDict(
		{
			"train": train_dataset,
			"valid": valid_dataset,
		}
	)
	print(raw_datasets)
	for key in raw_datasets["train"][0]:
		print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

	# Set up a context length as well as load a pretrained (GPT-2)
	# tokenizer.
	context_length = 256
	pretrained = "gpt2-medium"
	cache_dir = "./gpt2-medium-pretrained"
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
		raw_datasets["train"][:2]["contents"],
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
			element["contents"],
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

	# Load pretrained (GPT-2) model.
	config = GPT2Config.from_pretrained(
		pretrained,
		cache_dir=cache_dir,
		vocab_size=len(tokenizer),
		n_ctx=context_length,
		bos_token_id=tokenizer.bos_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)
	model = GPT2LMHeadModel(config)

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

	# Initialize the training arguments and the trainer.
	args = TrainingArguments(
		output_dir="./gpt2-medium-finetuned",
		per_device_train_batch_size=8,
		per_device_eval_batch_size=8,
		evaluation_strategy="steps",
		eval_steps=500,
		logging_steps=500,
		gradient_accumulation_steps=8,
		num_train_epochs=3,
		weight_decay=0.1,
		warmup_steps=100,
		lr_scheduler_type="cosine",
		learning_rate=5e-4,
		save_steps=500,
		#fp16=True, # Only use with CUDA devices.
	)

	trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=args,
		data_collator=data_collator,
		train_dataset=tokenized_datasets["train"],
		eval_dataset=tokenized_datasets["valid"],
	)

	# Train the model.
	trainer.train()
	trainer.save_model("./gpt2-medium-finetuned-final")

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