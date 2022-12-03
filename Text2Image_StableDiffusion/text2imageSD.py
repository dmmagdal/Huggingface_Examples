# text2imageSD.py
# Use the Stable Diffusion model from StabilityAI (download from
# HuggingFace) to create images from text.
# Source (Medium): https://towardsdatascience.com/how-to-generate-
#	images-from-text-with-stable-diffusion-models-ea9d1cb92f9b
# Source (Google Colab): https://colab.research.google.com/github/
#	huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
# Source (Huggingface Hub): https://huggingface.co/CompVis/stable-
#	diffusion-v1-4
# Source (GitHub): https://github.com/nicknochnack/StableDiffusionApp
# Source (YouTube): https://www.youtube.com/watch?v=7xc0Fs3fpCg
# Source (GitHub): https://github.com/huggingface/diffusers/tree/main/
#	src/diffusers/pipelines
# Windows/MacOS/Linux
# Python 3.7


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
	mps_device_available = torch.backends.mps.is_available()
	if mps_device_available:
		print("PyTorch detects mps device.")
	else:
		print("PyTorch does not detect mps device.")

	# Verify contents of saved model (local location). This is done by
	# computing the size of the folder. Note that the saved model is
	# about 2.5GB. Would also be valid to compute the hash of the
	# folder as well.
	saved_model = "./stable-diffusion-v1-4"
	load_saved = False
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
			print(f"{saved_model} is empty.")

	if load_saved:
		if cuda_device_available or mps_device_available:
			pipe = StableDiffusionPipeline.from_pretrained(
				saved_model,
				revision="fp16",
				torch_dtype=torch.float16,
			)
		else:
			pipe = StableDiffusionPipeline.from_pretrained(saved_model)
	else:
		# Initialize stable diffusion pipeline. This uses the V1.4
		# model weights.
		if cuda_device_available or mps_device_available:
			pipe = StableDiffusionPipeline.from_pretrained(
				"CompVis/stable-diffusion-v1-4", 
				revision="fp16",
				torch_dtype=torch.float16,
				use_auth_token=token, # pass token in to use it.
			)
		else:
			pipe = StableDiffusionPipeline.from_pretrained(
				"CompVis/stable-diffusion-v1-4", 
				use_auth_token=token, # pass token in to use it.
			)

		# Save a local copy of the model. Model is automatically cached to 
		# '~/.cache/huggingface/diffusers/models--CompVis--stable-diffusion
		# -v1-4' but since that cache may be cleared from time to time, it
		# is a good idea to keep a copy of the model here in the directory.
		pipe.save_pretrained(saved_model)

	if cuda_device_available:
		# Move pipeline to GPU.
		pipe = pipe.to("cuda")
	elif mps_device_available:
		pipe = pipe.to("mps")

	# Run inference with Pytorch's autocast module. There is some
	# variability to be expected in results, however there are also a
	# number of parameters that can be tweaked such as guidance_scale,
	# number_of_steps, and setting random seed (for deterministic
	# results) that should help get more consistent results.
	prompt = "A fighterjet flying over the desert"
	save = "diff1.png"
	if cuda_device_available:
		with autocast("cuda"):
			image = pipe(prompt).images[0]
	else:
		image = pipe(prompt).images[0]

	# Save image.
	image.save(save)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
