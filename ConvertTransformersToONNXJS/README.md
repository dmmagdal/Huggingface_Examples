# Convert Transformer To ONNX JS

Description: This folder seeks to explore to what extent HuggingFace transformers models and tokenizers can be used to when converted to ONNX for NodeJS for different tasks.


This project will look primarily at the BERT, GPT-2, and GPT-Neo models and tokenizers in particular, for the task of creating text embeddings (BERT) and text generation (GPT-2 and GPT-Neo). These tasks are relevant to the BERT_Database repository I have created with the overall task of replicating the BERT KNN database for RETRO in JS to make my implementation of the RETRO model mobile.


### Notes:
 - huggingface tokenizers npm library is not compatible with my current node version and could not be installed. Development is stalled on the ONNX-runtime nodejs side.


### Resources

 - GitHub example for [deploying a Huggingface BERT model to a website](https://github.com/jobergum/browser-ml-inference)
 - Huggingface tokenizers [npm package](https://www.npmjs.com/package/tokenizers)
 - GitHub to [Huggingface Tokenizers bindings in NodeJS](https://github.com/huggingface/tokenizers/tree/main/bindings/node)
 - [Relevant GitHub Issues post](https://github.com/huggingface/tokenizers/issues/491) to the Huggingface Tokenizers repo
 - Huggingface transformers [BERT](https://huggingface.co/docs/transformers/model_doc/bert) documentation
 - Huggingface transformers [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) documentation
 - Huggingface transformers [GPTNeo](https://huggingface.co/docs/transformers/model_doc/gpt_neo) documentation
 - Huggingface blog for [Faster Text Generation with Tensorflow & XLA](https://huggingface.co/blog/tf-xla-generate)
 - Huggingface transformers [Create a Custom Pipeline](https://huggingface.co/docs/transformers/add_new_pipeline) documentation
 - Huggingface transformers [Export to ONNX](https://huggingface.co/docs/transformers/serialization) documentation