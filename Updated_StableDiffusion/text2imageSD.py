# text2imageSD.py
# Run the new "updated" versions of Stable Diffusion on the 3 main
# image synthesis tasks (text-to-image, image-to-image with text-
# guidance, image inpainting).
# Windows/MacOS/Linux
# Python 3.7


import gc
import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


def main():
	# Load huggingface hub user access token (required to download
	# model).
	if os.path.exists(".env") and os.path.isfile(".env"):
		with open(".env", "r") as f:
			token = f.read().strip("\n")
	else:
		print("Missing .env file with Huggingface Hub user access token.")
		exit(0)

	# Torch cuda is required to run the stable diffusion model. Will
	# investigate alternative implementations or repos to run the model
	# on cpu.
	cuda_device_available = torch.cuda.is_available()
	if cuda_device_available:
		print("PyTorch detects cuda device.")
	else:
		print("PyTorch does not detect cuda device.")

	# A collection of models to try. Each key is the model name on
	# huggingface hub and the value is a list containing the following:
	# the local save name for the model, the output save file name, and
	# whether the model has a floating point 16 revision available.
	# Note that having a floating point 16 revision for the model does
	# NOT disqualify it from being loading torch.float16. Loading the
	# models in torch.float32 (default) is too large to load on 8GB
	# 2060 SUPER GPU and should be run on CPU.
	saved_models = {
		"CompVis/stable-diffusion-v1-4": [ # Stable Diffusion "v1" model
			"stable-diffusion-v1-4",
			"stable-diffusionv1-4-output.png",
			False
		],
		"runwayml/stable-diffusion-v1-5": [
			"stable-diffusion-v1-5",
			"stable-diffusionv1-5-output.png",
			False
		],
		"stabilityai/stable-diffusion-2": [ # Stable Diffusion "v2" model
			"stable-diffusion-v2-0",
			"stable-diffusionv2-output.png",
			False
		]
	}

	# Iterate through each model and run a quick text to image demo
	# with the associated prompt.
	for model, args in saved_models.items():
		# Unpack values for the model.
		saved_model, output_path, fp16_rev = args
		saved_model = "./" + saved_model
		load_saved = False

		# Verify contents of saved model (local location). This is done
		# by computing the size of the folder. Note that each saved
		# model is about 2.5GB. Would also be valid to compute the hash
		# of the folder as well.
		if os.path.exists(saved_model) and os.path.isdir(saved_model):
			# Calculate size of saved model folder. See reference:
			# https://www.geeksforgeeks.org/how-to-get-size-of-folder-
			# using-python/
			size = 0
			for path, dirs, files in os.walk(saved_model):
				for f in files:
					# Get size of file in path.
					fp = os.path.join(path, f)
					size += os.path.getsize(fp)

			# print(f"Folder size: {size}")
			if size != 0:
				load_saved = True
				print(f"{saved_model} validated as non-empty.")
			else:
				load_saved = False
				print(f"{saved_model} is empty.")

		# Load the model, either from a local copy or a download a copy
		# from huggingface hub.
		if load_saved:
			if cuda_device_available:
				pipe = StableDiffusionPipeline.from_pretrained(
					saved_model,
					revision="fp16" if fp16_rev else None,
					torch_dtype=torch.float16,
				)
			else:
				pipe = StableDiffusionPipeline.from_pretrained(
					saved_model
				)
		else:
			# Initialize model on the stable diffusion pipeline.
			if cuda_device_available:
				pipe = StableDiffusionPipeline.from_pretrained(
					model,
					revision="fp16" if fp16_rev else None,
					torch_dtype=torch.float16,
					use_auth_token=token, # pass token in to use it.
				)
			else:
				pipe = StableDiffusionPipeline.from_pretrained(
					model, 
					use_auth_token=token, # pass token in to use it.
				)

			# Save a local copy of the model. Model is automatically
			# cached to '~/.cache/huggingface/diffusers/models--{model}'
			# but since that cache may be cleared from time to time, it
			# is a good idea to keep a copy of the model here in the
			# directory.
			pipe.save_pretrained(saved_model)

		if cuda_device_available:
			# Move pipeline to GPU.
			pipe = pipe.to("cuda")

			if model == "stabilityai/stable-diffusion-2":
				# Use this function after sending the pipeline to cuda
				# (stable diffusion v2 only) to use less VRAM at the
				# cost of speed (This is for low GPU RAM). 
				pipe.enable_attention_slicing()

		# Run inference with Pytorch's autocast module. There is some
		# variability to be expected in results, however there are also a
		# number of parameters that can be tweaked such as guidance_scale,
		# number_of_steps, and setting random seed (for deterministic
		# results) that should help get more consistent results.
		prompt = "A fighterjet flying over the desert"
		if cuda_device_available:
			with autocast("cuda"):
				image = pipe(prompt).images[0]
		else:
			image = pipe(prompt).images[0]

		# Save image.
		image.save(output_path)

		# Delete the pipeline instance and clear the memory.
		del pipe
		gc.collect()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()