# text2imageWD.py
# Use the Waifu Diffusion (Stable Diffusion branch) model from hakurei
# (download from HuggingFace) to create anime images from text.
# Source (Google Colab): https://colab.research.google.com/drive/
#	1_8wPN7dJO746QXsFnB09Uq2VGgSRFuYE#scrollTo=1HaCauSq546O
# Source (Huggingface Hub): https://huggingface.co/hakurei/
#	waifu-diffusion
# Source (Huggingface Model Card): https://huggingface.co/hakurei/
#	waifu-diffusion-v1-3
# Source (GitHub Gist Release Notes): https://gist.github.com/harubaru/
#	f727cedacae336d1f7877c4bbe2196e1
# Source (GitHub): https://github.com/harubaru/waifu-diffusion
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

	# Verify contents of saved model (local location). This is done by
	# computing the size of the folder. Note that the saved model is
	# about 2.5GB. Would also be valid to compute the hash of the
	# folder as well.
	saved_model = "./waifu-diffusion-v1-3"
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
		if cuda_device_available:
			pipe = StableDiffusionPipeline.from_pretrained(
				saved_model,
				revision="fp16",
				torch_dtype=torch.float16,
			)
		else:
			pipe = StableDiffusionPipeline.from_pretrained(saved_model)
	else:
		# Initialize waifu diffusion on the stable diffusion pipeline.
		# This uses the V1.3 model weights.
		if cuda_device_available:
			pipe = StableDiffusionPipeline.from_pretrained(
				"hakurei/waifu-diffusion", 
				revision="fp16",
				torch_dtype=torch.float16,
				use_auth_token=token, # pass token in to use it.
			)
		else:
			pipe = StableDiffusionPipeline.from_pretrained(
				"hakurei/waifu-diffusion", 
				use_auth_token=token, # pass token in to use it.
			)

		# Save a local copy of the model. Model is automatically cached to 
		# '~/.cache/huggingface/diffusers/models--hakurei--waifu-diffusion'
		# but since that cache may be cleared from time to time, it
		# is a good idea to keep a copy of the model here in the directory.
		pipe.save_pretrained(saved_model)

	if cuda_device_available:
		# Move pipeline to GPU.
		pipe = pipe.to("cuda")

	# Run inference with Pytorch's autocast module. There is some
	# variability to be expected in results, however there are also a
	# number of parameters that can be tweaked such as guidance_scale,
	# number_of_steps, and setting random seed (for deterministic
	# results) that should help get more consistent results.
	prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed " +\
		"mouth, earrings, green background, hat, hoop earrings, " +\
		"jewelry, looking at viewer, shirt, short hair, simple " +\
		"background, solo, upper body, yellow shirt"
	save = "diff1.png"
	if cuda_device_available:
		with autocast("cuda"):
			image = pipe(prompt)["sample"][0]
	else:
		image = pipe(prompt)["sample"][0]

	# Save image.
	image.save(save)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()