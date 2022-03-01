# finetune_pretrained.py
# Walk through the Huggingface documentation tutorial on fine-tuning a
# pre-trained model.
# Source: https://huggingface.co/docs/transformers/training
# Source (Huggingface Datasets documentation): 
# https://huggingface.co/docs/datasets/
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import numpy as np
import tensorflow as tf
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import TFAutoModelForSequenceClassification
from transformers import get_scheduler
from tqdm.auto import tqdm


def main():
	# This tutorial will show how to fine-tune a pretrained model from
	# the Transformers library. In Tensorflow, models can be directly
	# trained using Keras and the fit() method. In PyTorch, there is no
	# generic training loop so the huggingface Transformers library
	# provides an API with the class Trainer allow for fine-tuning or
	# training a model from scratch easily. Will then show how to
	# alternatively write the whole training loop in PyTorch.
	# Before we can fine-tune the model, we need a dataset. In this
	# tutorial, we will show you how to fine-tune BERT on the IMDB
	# dataset: the task is to classify whether movie reviews are
	# positive or negative. For examples of other tasks, refer to the
	# additional-resources section!

	# Preparing the datasets
	# We will use the Huggingface Datasets library to download and
	# preprocess the IMDB datasets. Refer to the Huggingface Datasets
	# documentation (https://huggingface.co/docs/datasets/) or the
	# preprocessing (https://huggingface.co/docs/transformers/preprocessing)
	# tutorial for more information.

	# First, use load_dataset function to download and cache the
	# dataset.
	raw_datasets = load_dataset("imdb", cache_dir=".")

	# This works like the from_pretrained() method we saw for the
	# models and tokenizers (except the cache directory is
	# ~/.cache/huggingface/dataset by defualt). The raw_datasets object
	# is a dictionary with three keys: "train", "test", and
	# "unsupervised" (which correspond to the three splits of that
	# dataset). We will use the "train" split for training and the
	# "test" split for validation.

	# To preprocess the data, we will need a tokenizer.
	tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=".")


	# As we saw in preprocessing, we can prepate the text inputs for
	# the model with the following command (this is an example, not a
	# command you can execute):
	# inputs = tokenizer(
	#	sentences, padding="max_length", truncation=True
	# )
	# This would make all the samples have the maximum length of the
	# model can accept (here 512), either by padding or truncating
	# them.
	# However, we can instead apply these preprocessing steps to all
	# the splits of our dataset at once by using the map() method:
	def tokenize_function(examples):
		return tokenizer(
			examples["text"], padding="max_length", truncation=True
		)


	tokenized_datasets = raw_datasets.map(
		tokenize_function, batched=True
	)

	# You can learn more about the map() method or the other ways to
	# preprocess the data in the Huggingface Datasets documentation
	# (https://huggingface.co/docs/datasets/).
	# Next, we wil generate a small subset of the training and 
	# validation dataset, to enable faster training:
	small_train_dataset = tokenized_datasets["train"]\
		.shuffle(seed=42)\
		.select(range(1000))
	small_eval_dataset = tokenized_datasets["test"]\
		.shuffle(seed=42)\
		.select(range(1000))
	full_train_dataset = tokenized_datasets["train"]
	full_eval_dataset = tokenized_datasets["test"]

	# In all examples below, we will use small_train_dataset and
	# small_eval_dataset. Just replace them by their full
	# equivalent to train or evaluate on the full dataset.

	# Fine-tuning in PyT-rch with the Trainer API
	# Since PyTorch does not provide a training loop, the
	# Huggingface Transformers library provides a Trainer APU that is
	# optimized for Huggingface Transformers models, with a wide range
	# of training options and with built-in features like logging,
	# gradient accumulation, and mixed precision.

	# First, let's define the model.
	model = AutoModelForSequenceClassification.from_pretrained(
		"bert-base-cased"
	)

	# This will issue a warning about some of the pre=trained weights
	# not being used and some weights being randomly initialized.
	# That's because we are throwing away the pre-training head of the
	# BERT model to replace it with a classification head which is
	# randomly initialized. We will fine-tune this model on our task,
	# transfering the knowledge of the pretrained model to it (which is
	# why doing this is called transfer learning).
	# Then, to define our Trainer, we will need to instantiate a
	# TrainingArguments. This class contains all the hyperparameters we
	# can tune for the Trainer or the flags to activate the different
	# training options it supports. Let's begin by using all the
	# defaults, the only thing we then have to provide is a directory
	# in which the checkpoints will be saved.
	training_args = TrainingArguments("test_trainer")

	# Then we can instantiate a Trainer like this:
	trainer = Trainer(
		model=model, args=training_args,
		train_dataset=small_train_dataset, 
		eval_dataset=small_eval_dataset
	)

	# To fine-tune the model, just call train():
	trainer.train()

	# This will start a training that you can follow with a progress
	# bar, which should take a couple of minutes to complete (as long
	# as you have access to a GPU). It wont actually tell you anything
	# useful about how well (or badly) your model is performing however
	# as by default, there is no evaluation during training, and we
	# didn't tell the Trainer to compute any metrics. Let's have a look
	# on how to do that.
	# To have the Trainer comput and report metrics, we need to give it
	# a compute_metrics() function that takes predictions and labels
	# grouped in a namedtuple called EvalPrediction) and return a
	# dictionary with string items (the metric names) and float values
	# (the metric values).
	# The Huggingface Datasets library provides and easy way to get the
	# common metrics used in NLP with the load_metric() function. Here,
	# we simply use accuracy. Then, we define the compute_metrics()
	# function that just converts logits to predictions (remember that
	# all Huggingface Transformers models return the logits) and feed
	# them to compute() method of this metric.
	metric = load_metric("accuracy")


	def compute_metrics(eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		return metric.compute(
			predictions=predictions, references=labels
		)


	# The compute function needs to recieve a tuple (with logits and 
	# labels) and has to return a dictionary with string keys (the name
	# of the metric) and float values. It will be called at the end of
	# each evaluation phase on the whole arrays of predictions/labels.
	# To check if this works on practice, let's create a new Trainer
	# with our fine-tuned model:
	trainer = Trainer(
		model=model, args=training_args,
		train_dataset=small_train_dataset,
		eval_dataset=small_eval_dataset,
		compute_metrics=compute_metrics
	)
	trainer.evaluate()

	# Which showed an accuracy of around 87.5% in our case. If you want
	# to fine-tune your model and regularly report the evaluation
	# metrics (for instance at the end of each epoch), here is how you
	# should define your training arguments.
	training_args = TrainingArguments(
		"test_trainer", evaluation_strategy="epoch"
	)

	# See the documentation of TrainingArguments for more options.

	# Fine-tuning with Keras
	# Models can also be trained natively in Tensorflow using the Keras
	# API. First, let's define our model:
	model = TFAutoModelForSequenceClassification

	# Then we will need to convert our datasets from before to
	# tf.data.Dataset. Since we have fixed shapes, it can easily be
	# done like this. First, we remove the "text" column from our
	# datasets and set them in Tensorflow format:
	tf_train_dataset = small_train_dataset.remove_columns(["text"])\
		.with_format("tensorflow")
	tf_eval_dataset = small_eval_dataset.remove_columns(["text"])\
		.with_format("tensorflow")

	# Then convert everything in big tensors and use the
	# tf.data.Dataset.from_tensor_slices() method.
	train_features = {x:tf_train_dataset[x] for x in tokenizer.model_input_names}
	train_tf_dataset = tf.data.Dataset.from_tensor_slices(
		(train_features, tf_train_dataset["label"])
	)
	train_tf_dataset = train_tf_dataset.shuffle(len(tf_train_dataset))\
		.batch(8)

	eval_features = {x:tf_eval_dataset[x] for x in tokenizer.model_input_names}
	eval_tf_dataset = tf.data.Dataset.from_tensor_slices(
		(eval_features, tf_eval_dataset["label"])
	)
	eval_tf_dataset = eval_tf_dataset.shuffle(len(tf_eval_dataset))\
		.batch(8)

	# With this done, the model can be compiled and trained as any
	# Keras model.
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=tf.metrics.SparseCategoricalAccuracy(),
	)

	model.fit(train_tf_dataset, eval_tf_dataset, epochs=3)

	# With the tight interoperability between Tensorflow and PyTorch
	# models, you can even save the model and then reload it as a
	# PyTorch model (or vice-versa).
	model.save_pretrained("my_imdb_model")
	pytorch_model = AutoModelForSequenceClassification.from_pretrained(
		"my_imdb_model", from_tf=True
	)

	# Fine-tuning in native PyTorch
	# You might want to execute the following code at this stage to
	# free up memory.
	del model
	del pytorch_model
	del trainer
	torch.cuda.empty_cache()

	# Now let's see how to achieve the same results as in trainer
	# section in PyTorch. First, we need to define the dataloaders,
	# which we will use to iterate over batches. We just need to apply
	# a bit of post-processing to our tokenized_datasets before doing
	# that to:
	# -> remove the columns corresponding to values the model does not
	#	expect (here the "text" column).
	# -> rename the column "label" tp "labels" (because the model
	#	expects the argument to be named labels).
	# -> set the format of the datasets so they return PyTorch tensors
	#	instead of lists.
	# Our tokenized_datasets has one method for each of those steps:
	tokenized_datasets = tokenized_datasets.remove_columns(["text"])
	tokenized_datasets = tokenized_datasets.rename_column(
		"label", "labels"
	)
	tokenized_datasets.set_format("torch")

	small_train_dataset = tokenized_datasets["train"]\
		.shuffle(seed=42)\
		.select(range(1000))
	small_eval_dataset = tokenized_datasets["test"]\
		.shuffle(seed=42)\
		.select(range(1000))

	# Now that this is done, we can easily define the dataloaders.
	train_dataloader = DataLoader(
		small_train_dataset, shuffle=True, batch_size=8
	)
	eval_dataloader = DataLoader(
		small_eval_dataset, batch_size=8
	)

	# Define the model.
	model = AutoModelForSequenceClassification.from_pretrained(
		"bert-base-cased", num_labels=2
	)

	# Almost ready to write the training loop, the only two things that
	# are missing are an optimizer and a learning rate scheduler. The
	# default optimizer used by the Trainer is AdamW.
	optimizer = AdamW(model.parameters(), lr=5e-5)

	# Finally, the learning rate scheduler used by default is just a
	# linear decay from the maximum value (5e-5) to 0:
	num_epochs = 3
	num_training_steps = num_epochs * len(train_dataloader)
	lr_scheduler = get_scheduler(
		"linear", optimizer=optimizer, num_warmup_steps=0,
		num_training_steps=num_training_steps
	)

	# One last thing, we will want to use a GPU if we have access to
	# one (otherwise training might take several hours instead of a
	# couple of minutes). To do this, define a device we will put our
	# model and our batches on.
	device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
	model.to(device)

	# Now we are ready to train. To get some sense of when it will be
	# finished, we add a progress bar over our number of training steps
	# using the tqdm library.
	progress_bar = tqdm(range(num_training_steps))

	model.train()
	for epoch in range(num_epochs):
		for batch in train_dataloader:
			batch = {k:v.to(device) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()

			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad()
			progress_bar.update(1)

	# Note that if you are used to freezing the body of your
	# pre-trained model (like in computer vision) the above may seem a
	# bit strange, as we are directly fine-tuning the whole model
	# without taking any precaution. It actually works better this way
	# for Transformers model (so this is not an oversight on our side).
	# If you're not familiar with what "freezing the body" of the model
	# means, forget you read this paragraph.
	# Now to check the results, we need to write the evaluation loop.
	# Like in the trainer section, we will use a metric from the
	# Datasets library. Here we accumulate the predictions at each 
	# batch before computing the final result when the loop is
	# finished.
	metric = load_metric("accuracy")
	model.eval()
	for batch in eval_dataloader:
		batch = {k:v.to(device) for k, v in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)

		logits = outputs.logits
		predictions = torch.argmax(logits, dim=-1)
		metric.add_batch(
			predictions=predictions, references=batch["labels"]
		)
	metric.compute()

	# Additional Resources
	# To look at more fine-tuning examples you can refer to:
	# -> Huggingface Transfomers Examples 
	#	(https://github.com/huggingface/transformers/tree/master/examples) 
	#	which includes scripts to train on all common NLP tasks in
	#	PyTorch and Tensorflow.
	# ->Huggingface Transfomers Notebooks
	#	(https://huggingface.co/docs/transformers/notebooks) which 
	#	contains various notebooks and in particular one per task (look
	#	for the "how to finetune a model on xxx").

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()