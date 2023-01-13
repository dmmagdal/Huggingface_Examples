# custom_pipeline.py
# author: Diego Magdaleno
# Create a custom pipeline 
# Source (huggingface): https://huggingface.co/docs/transformers/add_
#   new_pipeline
# Python 3.7
# Windows/MacOS/Linux


import torch
from transformers import BertTokenizer, BertModel
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

	# Embed text to 
	input_text = "Hello there!"
	encoded_input = bert_tokenizer.encode(
		input_text, add_special_tokens=True
	)
	encoded_input = torch.tensor([encoded_input])

	# BERT's model output is a tuple of two outputs. The first output
	# is the sequence output of shape (batch_size, max_len, 
	# hidden_state) while the second output is the pooled output of 
	# shape (batch_size, hidden_state) where hidden_state = 768.
	last_hidden_states = bert_model(encoded_input)
	sequence_output, pooled_output = last_hidden_states


	# Create new pipeline.
	# First and foremost, you need to decide the raw entries the 
	# pipeline will be able to take. It can be strings, raw bytes, 
	# dictionaries or whatever seems to be the most likely desired 
	# input. Try to keep these inputs as pure Python as possible as it 
	# makes compatibility easier (even through other languages via 
	# JSON). Those will be the inputs of the pipeline (preprocess).
	#
	# Then define the outputs. Same policy as the inputs. The simpler, 
	# the better. Those will be the outputs of postprocess method.
	#
	# Start by inheriting the base class Pipeline. with the 4 methods 
	# needed to implement preprocess, _forward, postprocess and 
	# _sanitize_parameters.
	class BERTEmbeddingsPipeline(Pipeline):
		# _sanitize_parameters exists to allow users to pass any 
		# parameters whenever they wish, be it at initialization 
		# time pipeline(...., maybe_arg=4) or at call time pipe = 
		# pipeline(...); output = pipe(...., maybe_arg=4).
		#
		# The returns of _sanitize_parameters are the 3 dicts of 
		# kwargs that will be passed directly to preprocess, 
		# _forward and postprocess. Don’t fill anything if the 
		# caller didn’t call with any extra parameter. That allows to 
		# keep the default arguments in the function definition which 
		# is always more “natural”.
		def _sanitize_parameters(self, **kwargs):
			preprocess_kwargs = {}
			if "maybe_arg" in kwargs:
				preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
			return preprocess_kwargs, {}, {}

		# preprocess will take the originally defined inputs, and turn 
		# them into something feedable to the model. It might contain 
		# more information and is usually a Dict.
		def preprocess(self, inputs, maybe_arg=2):
			model_input = torch.Tensor(inputs["input_ids"])
			return {"model_input": model_input}

		# _forward is the implementation detail and is not meant to be 
		# called directly. forward is the preferred called method as it
		# contains safeguards to make sure everything is working on the
		# expected device. If anything is linked to a real model it 
		# belongs in the _forward method, anything else is in the 
		# preprocess/postprocess.
		def _forward(self, model_inputs):
			# model_inputs == {"model_input": model_input}
			outputs = self.model(**model_inputs)
			# Maybe {"logits": Tensor(...)}
			return outputs

		# postprocess methods will take the output of _forward and turn
		# it into the final output that were decided earlier.
		def postprocess(self, model_outputs):
			# best_class = model_outputs["logits"].softmax(-1)
			# return best_class
			return model_outputs[1]


	# Add to list of supported (pipeline) tasks.
	PIPELINE_REGISTRY.register_pipeline(
		'bert-embeddings',
		pipeline_class=BERTEmbeddingsPipeline,
		# pt_model=bert_model
	)

	emb_pipeline = pipeline(
		'bert-embeddings', 
		tokenizer=bert_tokenizer, 
		model=bert_model_name
	)

	print(emb_pipeline(input_text))
	# feature_extraction = pipeline(
	# 	'feature-extraction', model=bert_model, tokenizer=bert_tokenizer
	# )
	# output = feature_extraction(input_text)
	# print(output[0].shape)
	# print(feature_extraction(input_text))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
