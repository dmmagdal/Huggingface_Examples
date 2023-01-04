# text2imageMiscSD.py
# Use the miscellaneous stable diffusion models finetuned in
# different styles to create images from text.
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

	# Save outputs folder for images.
	output_dir = "./outputs"
	if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
		print("Ouptuts directory does not exist. Creating it now...")
		os.makedirs(output_dir)

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

	# A collection of models to try. Each key is the model name on
	# huggingface hub and the value is a list containing the following:
	# the local save name for the model, the test input prompt, the
	# output save file name, and whether the model has a floating point
	# 16 revision available. Note that having a floating point 16
	# revision for the model does NOT disqualify it from being loading
	# torch.float16. Loading the models in torch.float32 (default) is
	# too large to load on 8GB 2060 SUPER GPU and should be run on CPU.
	saved_models = {
		"dallinmackay/Tron-Legacy-diffusion": [
			"tron-legacy-diffusion", 
			"city landscape in the style of trnlgcy",
			"tron-landscape.png",
			False
		],
		"nousr/robo-diffusion": [
			"robo-diffusion",
			"portrait of nousr robot photorealistic, trending on artstation",
			"robo-portrait.png",
			False
		],
		"nitrosocke/classic-anim-diffusion": [
			"classic-anim-diffusion",
			"classic disney style magical princess with golden hair",
			"classic-disney-princess.png",
			False
		],
		"nitrosocke/archer-diffusion": [
			"archer-diffusion",
			"a magical princess with golden hair, archer style",
			"archer-princess.png",
			False
		],
		"nitrosocke/spider-verse-diffusion": [
			"spider-verse-diffusion",
			"a magical princess with golden hair, spiderverse style",
			"spiderverse-princess.png",
			False
		],
		"nitrosocke/elden-ring-diffusion": [
			"elden-ring-diffusion",
			"a magical princess with golden hair, elden ring style",
			"elden-ring-princess.png",
			False
		],
		"nitrosocke/mo-di-diffusion": [
			"mo-di-diffusion",
			"a magical princess with golden hair, modern disney style",
			"modern-disney-princess.png",
			False
		],
		"nitrosocke/Arcane-Diffusion": [
			"arcane-diffusion",
			"arcane style, a magical princess with golden hair",
			"arcane-princess.png",
			False
		],
		"lambdalabs/sd-pokemon-diffusers": [
			"sd-pokemon-diffusers",
			"Yoda",
			"yoda-pokemon.png",
			False
		],
		"hakurei/waifu-diffusion": [
			"waifu-diffusion",
			"1girl, aqua eyes, baseball cap, blonde hair, closed "+\
				"mouth, earrings, green background, hat, hoop "+\
				"earrings, jewelry, looking at viewer, shirt, short "+\
				"hair, simple background, solo, upper body, yellow "+\
				"shirt",
			"blonde-girl-waifu.png",
			True
		],
		"DGSpitzer/Cyberpunk-Anime-Diffusion": [
			"cyberpunk-anime-diffusion",
			"a beautiful perfect face girl in dgs illustration "+\
				"style, Anime fine details portrait of school girl "+\
				"in front of modern tokyo city landscape on the "+\
				"background deep bokeh, anime masterpiece, 8k, "+\
				"sharp high quality anime",
			"cyberpunk-anime-girl.png",
			True
		],
		"nitrosocke/redshift-diffusion": [
			"redshift-diffusion",
			"redshift style robert downey jr as ironman",
			"redshift-ironman.png",
			False
		],
		"nitrosocke/Ghibli-Diffusion": [
			"ghibli-diffusion",
			"ghibli style magical princess with golden hair",
			"ghibli-princess.png",
			False
		],
		"prompthero/openjourney": [
			"midjourney-v4-diffusion",
			"retro serie of different cars with different colors "+\
				"and shapes, mdjrny-v4 style",
			"midjourney-v4-cars.png",
			False
		],
		"Aybeeceedee/knollingcase": [
			"knollingcase-diffusion",
			"(clockwork:1.2), knollingcase, labelled, overlays, "+\
				"oled display, annotated, technical, knolling "+\
				"diagram, technical drawing, display case, dramatic "+\
				"lighting, glow, dof, reflections, refractions",
			"knollingcase-clockwork.png",
			False
		],
		"Linaqruf/anything-v3.0": [
			"anything-v3-diffusion",
			"1girl, white hair, golden eyes, beautiful eyes, "+\
				"detail, flower meadow, cumulonimbus clouds, "+\
				"lighting, detailed sky, garden",
			"anything-v3-anime-girl.png",
			False
		],
		"Envvi/Inkpunk-Diffusion": [
			"inkpunk-diffusion",
			"(nvinkpunk), portrait of a girl, perfect female face, "+\
				"intricate, highly detailed, happy, digital "+\
				"painting, intense colors, (colorful), (high "+\
				"contrast colors), sharp focus, 8k, highly detailed",
			"inkpunk-girl.png",
			False
		],
		"dreamlike-art/dreamlike-diffusion-1.0": [
			"dreamlike-diffusion-v1",
			"dreamlikeart, a grungy woman with rainbow hair, "+\
				"travelling between dimensions, dynamic pose, happy, "+\
				"soft eyes and narrow chin, extreme bokeh, dainty "+\
				"figure, long hair straight down, torn kawaii shirt "+\
				"and baggy jeans, In style of by Jordan Grimmer and "+\
				"greg rutkowski, crisp lines and color, complex "+\
				"background, particles, lines, wind, concept art, "+\
				"sharp focus, vivid colors",
			"dreamlike-diffusion-v1.png",
			False
		],
		# "nerijs/isopixel-diffusion-v1": [ # finetuned from Stable Diffusion v2 
		# 	"isopixel-diffusion",
		# 	"isometric bedroom, isopixel style",
		# 	"isopixel-bedroom.png",
		# 	False
		# ]
		"nousr/robo-diffusion-2-base": [ # finetuned from Stable Diffusion v2 
			"robo-diffusion2",
			"A realistic photograph of a 3d nousr robot in a modern "+\
				"city. A glossy white and orange nousr robot.",
			"robo-portrait-sd2.png",
			False,
			"black and white robot, picture frame, a children's "+\
				"drawing in crayon. #Wholesale, Abstract Metal "+\
				"Sculpture. i'm leaving a bad review."
		],
		# "hakurei/waifu-diffusion-v1-4": [ # finetuned from Stable Diffusion v2-1
		# 	"waifu-diffusion-v1-4",
		# 	"masterpiece, best quality, 1girl, green hair, sweater, "+\
		# 		"looking at viewer, upper body, beanie, outdoors, "+\
		# 		"watercolor, night, turtleneck",
		# 	"waifu-diffusion-v1-4.png",
		# 	False,
		# 	""
		# ]
	}

	# Iterate through each model and run a quick text to image demo
	# with the associated prompt.
	for model, args in saved_models.items():
		# Unpack values for the model.
		if len(args) == 4:
			saved_model, prompt, output_path, fp16_rev = args
			neg_prompt = None
		else:
			saved_model, prompt, output_path, fp16_rev, neg_prompt = args
		saved_model = "./" + saved_model
		output_path = os.path.join(output_dir, output_path)
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

		# Move pipeline to GPU (if applicable). Use attention slicing
		# function after sending the pipeline to the GPU to use less
		# VRAM at the cost of speed (This is for low GPU RAM).
		if cuda_device_available:
			# Move pipeline to GPU.
			pipe = pipe.to("cuda")
			pipe.enable_attention_slicing()
		elif mps_device_available:
			pipe = pipe.to("mps")
			pipe.enable_attention_slicing()

		# Run inference with Pytorch's autocast module. There is some
		# variability to be expected in results, however there are also a
		# number of parameters that can be tweaked such as guidance_scale,
		# number_of_steps, and setting random seed (for deterministic
		# results) that should help get more consistent results.
		if cuda_device_available:
			if len(args) == 4:
				with autocast("cuda"):
					image = pipe(prompt).images[0]
			else:
				with autocast("cuda"):
					image = pipe(
						prompt, negative_prompt=neg_prompt
					).images[0]
		else:
			if len(args) == 4:
				image = pipe(prompt).images[0]
			else: 
				image = pipe(prompt, negative_prompt=neg_prompt).images[0]

		# Save image.
		image.save(output_path)

		# Delete the pipeline instance and clear the memory.
		del pipe
		gc.collect()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()