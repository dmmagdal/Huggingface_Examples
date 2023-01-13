# load_models.py
# author: Diego Magdaleno
# Load HuggingFace models and tokenizers in python and convert them to
# ONNX format so that they can be read in NodeJS.
# Source (BERT embeddings): https://discuss.huggingface.co/t/how-to-
#	get-embedding-matrix-of-bert-in-hugging-face/10261/2
# Python 3.7
# Windows/MacOS/Linux


import os
import json
import shutil
import torch
import transformers
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from transformers import AutoModelForCausalLM #, GPT2Model
from transformers import convert_graph_to_onnx as onnx_convert
from transformers import pipeline, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY


def main():
	# Load tokenizer and model for BERT.
	bert_model_name = 'bert-base-uncased'
	bert_tokenizer = BertTokenizer.from_pretrained(
		bert_model_name
	)
	bert_model = BertModel.from_pretrained(
		bert_model_name
	)

	# Convert text to BERT embeddings.
	input_text = "Hello there!"
	encoded_input = bert_tokenizer.encode(
		input_text, add_special_tokens=True
	)
	encoded_input = torch.tensor([encoded_input])

	# BERT's model output is a tuple of two outputs. The first output
	# is the sequence output of shape (batch_size, max_len, 
	# hidden_state) while the second output is the pooled output of 
	# shape (batch_size, hidden_state) where hidden_state = 768.
	# with torch.no_grad():
	# 	last_hidden_states = bert_model(encoded_input)[0] # Models outputs are now tuples
	# last_hidden_states = last_hidden_states.mean(1)
	last_hidden_states = bert_model(encoded_input)[1] # Model output is tuples.
	print(f"Input: {input_text}")
	print(f"BERT hidden state (embedding): {last_hidden_states}, {last_hidden_states.shape}") # size of last_hidden_states is [1,768]
	
	# Load tokenizer and model for GPT-2.
	gpt2_model_name = "gpt2"

	# You can use either the GPT2Tokenizer OR the AutoTokenizer.
	gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
		gpt2_model_name
	)
	# gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)

	# There are several ways to initialize the GPT2 model (for text
	# generation), Any of these (except GPTModel) work. With GPT2Model,
	# that model wont work in generating text because GPT2Model is just
	# a generic model without a language model head.
	# gpt2_model = GPT2Model.from_pretrained( 
	# 	gpt2_model_name
	# )
	gpt2_model = GPT2LMHeadModel.from_pretrained(
		gpt2_model_name
	)
	# gpt2_model = TFAutoModelForCausalLM.from_pretrained(
	# 	gpt2_model_name
	# )
	# gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)

	# Note that for using the TFAutoModelForCausalLM, the tokenizer
	# MUST return tensorflow tensors
	encoded_input = gpt2_tokenizer([input_text], return_tensors='pt')
	# encoded_input = gpt2_tokenizer([input_text], return_tensors='tf')

	# Refer to the documentation for the various arguments/parameters
	# to specify for generating text.
	output = gpt2_model.generate(**encoded_input, do_sample=True)

	# Tokenizer can only decode one element at a time from the output batch.
	decoded_output = gpt2_tokenizer.decode(output[0])
	print(f"GPT-2 output: {decoded_output}")

	# Extract/save the vocabularies of the tokenizers. Note that BERT
	# only has a vocab.txt while GPT2 has a merges.txt and vocab.json
	# file.
	bert_tokenizer.save_vocabulary("./", "bert")
	gpt2_tokenizer.save_vocabulary("./", "gpt2")

	# Save tokenizers and models locally (in the same place/under the 
	# same name) and use the following command to convert them to ONNX:
	# `python -m transformers.onnx --model=local-pt-checkpoint onnx/`
	bert_tokenizer.save_pretrained("BERT-base-uncased")
	bert_model.save_pretrained("BERT-base-uncased")
	gpt2_tokenizer.save_pretrained("GPT2")
	gpt2_model.save_pretrained("GPT2")



	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
