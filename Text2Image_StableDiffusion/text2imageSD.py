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
# from huggingface_hub import notebook_login


def main():
	# Enable GPU access and external widgets if running code in Colab
	# notebook.
	#from google.colab import output
	#output.enable_custom_widget_manager()

	# Required modules (from medium article)
	# diffusers==0.2.4 Changed to 0.3.0 from Nick's repo
	# ftfy
	# ipywidgets>=7,<8
	# pillow
	# scipy
	# torch
	# transformers

	# Load huggingface hub user access token (required to download
	# model).
	if os.path.exists(".env") and os.path.isfile(".env"):
		with open(".env", "r") as f:
			token = f.read().strip("\n")
	else:
		print("Missing .env file with Huggingface Hub user access token.")
		exit(0)

	# Log into huggingface with user token.
	# notebook_login()

	# Initialize stable diffusion pipeline. This uses the V1.4 model
	# weights.
	pipe = StableDiffusionPipeline.from_pretrained(
		"CompVis/stable-diffusion-v1-4", 
		revision="fp16",
		torch_dtype=torch.float16,
		use_auth_token=token, # pass token in to use it.
	)
	# pipe = StableDiffusionPipeline.from_pretrained(
	# 	"./stable-diffusion-v1-4"
	# )

	# Move pipeline to GPU.
	pipe = pipe.to("cuda")

	# Run inference with Pytorch's autocast module. There is some
	# variability to be expected in results, however there are also a
	# number of parameters that can be tweaked such as guidance_scale,
	# number_of_steps, and setting random seed (for deterministic
	# results) that should help get more consistent results.
	prompt = "A fighterjet flying over the desert"
	save = "diff1.png"

	with autocast("cuda"):
		image = pipe(prompt)["sample"][0]

	image.save(save)


if __name__ == '__main__':
	main()