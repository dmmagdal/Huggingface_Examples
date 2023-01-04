# Text to Image with Miscellaneous Stable Diffusion Models

Description: Leverage various fine-tuned diffusion models for generating art from text prompts.


### Requirements

 - The version of the diffusers library outlined in `requirements.txt` requires Python 3.7+.
 - The `ipywidgets>=7,<8` library is only required if running this code in a Google Colab environment (this allows for images to be displayed within the notebook).
 - Create a `.env` file containing your Huggingface Hub user access token. This token is required to download the stable diffusion model. You must also agree to the huggingface terms of service associated with downloading the model. If there is a permission error the first time running this, visit the help link provided in the message.


### Notes:

 - Refer to the following README.md Notes sections from previous Stable Diffusion examples for guidance on running Stable Diffusion (including text to image, image to image, and image inpainting): [Text2Image_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/WaifuDiffusion_StableDiffusion/README.md), [Image2Image_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/Image2Image_StableDiffusion/README.md), and [InPainting_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/InPainting_StableDiffusion/README.md) [WaifuDiffusion_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/WaifuDiffusion_StableDiffusion/README.md)
 - This example primarily deals with just text to image. However the [Finetuned Diffusion](https://huggingface.co/spaces/anzorq/finetuned_diffusion) model/demo shows that these models can be applied to image to image and inpainting applications.
 - The following models, **Inkpunk Diffusion** and **Knollingcase Diffusion**, require the `diffusers` module to be version `0.9.0`. All other models are able to run on `diffusers==0.3.0` and above.
 - With `diffusers==0.9.0`, there are additional stable diffusion pipeline arguments that enhance the experience of using these diffusion models (refer to [here](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion#diffusers.StableDiffusionPipeline) for more information):
    - `negative_prompt` allows for negative prompting input.
    - `num_images_per_prompt` allows for a user to specify the number of images they want output at a time.


### Models
 - **Tron Legacy Diffusion**
    - Use the token "trnlgcy" in prompts to use the style. ie "\[person\] in the style of trnlgcy" | steps: 25, sampler: "Euler a", CFG scale: 7.5
    - based on the v1.5 Stable Diffusion model (v1.5 refering to [runwayml's model](https://huggingface.co/runwayml/stable-diffusion-v1-5))
    - This model was trained with Dreambooth training by TheLastBen, using 30 images at 3000 steps.
 - **Robo Diffusion**
    - [Colab](https://colab.research.google.com/github/nousr/robo-diffusion/blob/main/robo_diffusion_v1.ipynb)
    - [GitHub](https://github.com/nousr/robo-diffusion)
    - Keep the words "nousr robot" towards the beginning of the prompt
 - **Classic Anim Diffusion**
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - This model was trained using the diffusers based dreambooth training by ShivamShrirao using prior-preservation loss and the "train-text-encoder" flag in 9,000 steps.
 - **Archer Diffusion**
    - Use the token "archer style" in prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - This model was trained using the diffusers based dreambooth training and prior-preservation loss in 4,000 steps and using the "train-text-encoder" feature.
 - **Spider-verse Diffusion**
    - Use the token "spiderverse style" in prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - This model was trained using the diffusers based dreambooth training and prior-preservation loss in 3,000 steps.
 - **Elden Ring Diffusion**
    - Use the token "elden ring style" in prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - This model was trained using the diffusers based dreambooth training and prior-preservation loss in 3,000 steps.
 - **Mo(dern) Di(sney) Diffusion**
    - Use the token "modern disney style" in prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - This model was trained using the diffusers based dreambooth training by ShivamShrirao using prior-preservation loss and the "train-text-encoder" flag in 9,000 steps.
    - [Colab](https://colab.research.google.com/drive/1j5YvfMZoGdDGdj3O3xRU1m4ujKYsElZO?usp=sharing)
 - **Arcane Diffusion**
    - Use the token "arcane style" in prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - [Colab](https://colab.research.google.com/drive/1j5YvfMZoGdDGdj3O3xRU1m4ujKYsElZO?usp=sharing)
    - Version 1: (arcane-diffusion-5k) This model was trained using "Unfrozen Model Textual Inversion" utilizing the "Training with prior-preservation loss" methods. There is still a slight shift towards the style, while not using the arcane token.
    - Version 2: (arcane-diffusion-v2) This uses the diffusers based dreambooth training and prior-preservation loss and is way more effective. The diffusers were then converted with a script to a ckpt file in order to work with automatic1111's repo. Training was done with 5k steps for a direct comparison to v1 and results show that it needs more steps for a more prominent result. Version 3 will be tested with 11k steps. 
    - Version 3: This version uses the new "train-text-encoder" setting and improves the quality and edibility of the model immensely. Trained on 95 images from the show in 8,000 steps.
 - **Pokemon Diffusion**
    - Trained on BLIP captioned Pokemon images using 2x A6000 GPUs on Lambda GPU Cloud for around 15,000 steps (6 hours, at a cost of about $10)
 - **Waifu Diffusion**
    - A latent text-to-image diffusion model that has been conditioned on high-quality anime images through fine-tuning.
    - [Colab](https://colab.research.google.com/drive/1_8wPN7dJO746QXsFnB09Uq2VGgSRFuYE#scrollTo=1HaCauSq546O)
 - **Cyberpunk Anime Diffusion**
    - Based off a finetuned Waifu Diffusion v1.3 model with Stable Diffusion v1.5 (again, the runwayml fork) new VAE, trained in Dreambooth.
    - Use keywords "dgs" in prompt, with "illustrated style" to get even better results. For sampler, user "Euler a" for best results (DDIM kinda works too), CFG scale: 7.5, steps 20 should be fine
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - [GitHub](https://github.com/HelixNGC7293/cyberpunk-anime-diffusion)
 - **Redshift Diffusion**
    - Use the tokens "redshift style" in prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - This model was trained using the diffusers based dreambooth training by ShivamShrirao using prior-preservation loss and the train-text-encoder flag in 11,000 steps.
 - **Ghibli Diffusion**
    - Use the tokens "ghibli style" in prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - This model was trained using the diffusers based dreambooth training by ShivamShrirao using prior-preservation loss and the train-text-encoder flag in 15,000 steps.
 - **Open Journey Diffusion**
    - Use the tokens "midjrny-v4 style" at the beginning of prompts for effect
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
    - [Colab](https://colab.research.google.com/drive/1vkuxKKeSYNYI2OLZm8mR-WqcokQtSURM?usp=sharing)
 - **Knollingcase Diffusion**
    - Use the token "knollingcase" anywhere in the prompt for effect
 - **Anything v3 Diffusion**
    - THis model is intended to produce high-quality, highly detailed anime style with just a few prompts. Like any other anime-style Stable Diffusion models, it also supports danbooru tags to generate images
    - Can export the model to ONNX, MPS, and/or FLAX/JAX
 - **Inkpunk Diffusion**
    - Use the token "nvinkpunk" in prompts for effect
    - Vaguely inspired by Gorillaz, FLCL, and Yoji Shinkawa
 - **IsoPixel Diffusion**
    - Use the token "isopixel" in prompts for effect
    - Stable Diffusion v2-768 model trained on to generate isometric pixel art
    - Always use 768 x 768
    - High step count on Euler_a gives best results
    - Low CFG scale outputs great results
    - Can use tools like Pixelator to achieve a better effect
    - Is currently only supported with .ckpt version (which means it cannot run with the pipeline)
 - **Robo Diffusion 2 (base)**
    - Use the token "nousr robot" towards the beginning of the prompt for effect
    - A dreambooth-method finetune of stable diffusion that will output cool looking robots when prompted
    - Use negative prompts to achieve best result
 - **Dreamlike Diffusion 1.0**
    - Use the same prompts as you would for SD 1.5. Add "dreamlikeart" if the artstyle is too weak.
    - Non-square aspect ratios work better for some prompts. If you want a portrait photo, try using a 2:3 or a 9:16 aspect ratio. If you want a landscape photo, try using a 3:2 or a 16:9 aspect ratio.
    - Use slightly higher resolution for better results: 640x640px, 512x768px, 768x512px, etc.
    - Dreamlike Diffusion 1.0 is SD 1.5 fine tuned on high quality art, made by [dreamlike.art](https://dreamlike.art/).
 - **Waifu Diffusion v1.4**
    - Based on Stable Diffusion v2.1
    - Generally an update of the previously mentioned Waifu Diffusion model hosted by the same person of Huggingface Hub


### Embeddings
 - There are different embeddings that exist thanks to the ability to fine-tune Stable Diffusion through a process called Textual Inversion. Textual inversion allows a user to train a tiny part of the network on their own pictures, and use the results when generating new ones. In this context, the embedding is the part of the neural network that is trained. These embeddings are usually a .pt or .bin file. For more information see [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Textual-Inversion).
 - **Action Helper**
    - Trained for 500 steps with a lr of 0.003 and 4 steps gradient accumulation.
 - **Cinema Helper**
    - Nice bokeh, grain, depth of field, soft lights, muted colors, and overall a great cinema vibe.
    - Trained for 1000 steps on 5 steps gradient accumulation and a lr of 0.003 for the first half, then 0.001 for the second half.
 - **Photo Helper**
    - Photohelper is a Stable Diffusion 2.x embedding with the goal of generating photorealistic pictures with nice colors. Its training data consists of ~120 of @spaablauw (on Huggingface) photos, half of which are portraits.
    - It works best if you play with the weight a bit, and add terms related to photography.
 - **Vintage Helper**
    - VintageHelper will make your Stable Diffusion 2.0/2.1 generations feel analog. It focuses on bokeh, colorgrading, depth of field, composition and adds grain and imperfection. It's been trained for 500 steps at a learning rate of 0.002, and the next 500 at a learning rate of 0.001; 15 steps gradient accumulation. All 104 training images were captioned with BLIP and corrected.
    - Included is the .pt for both 600 and 1000 steps, the ones @spaablauw feels are best.
 - **Knollingcase SD v2.0**
    - The embeddings in this repository were trained for the 768px Stable Diffusion v2.0 model. The embeddings should work on any model that uses SD v2.0 as a base.
    - Currently the kc32-v4-5000.pt & kc16-v4-5000.pt embeddings seem to perform the best.
    - Knollingcase v1: The v1 embeddings were trained for 4000 iterations with a batch size of 2, a text dropout of 10%, & 16 vectors using Automatic1111's WebUI. A total of 69 training images with high quality captions were used.
    - Knollingcase v2: The v2 embeddings were trained for 5000 iterations with a batch size of 4 and a text dropout of 10%, & 16 vectors using Automatic1111's WebUI. A total of 78 training images with high quality captions were used.
    - Knollingcase v3: The v3 embeddings were trained for 4000-6250 iterations with a batch size of 4 and a text dropout of 10%, & 16 vectors using Automatic1111's WebUI. A total of 86 training images with high quality captions were used.
    - Knollingcase v4: The v4 embeddings were trained for 4000-6250 iterations with a batch size of 4 and a text dropout of 10%, using Automatic1111's WebUI. A total of 116 training images with high quality captions were used.
    - Usage:
       - To use the embeddings, download and then rename the files to whatever trigger word you want to use. They were trained with kc8, kc16, kc32, but any trigger word should work.
       - The knollingcase style is considered to be a concept inside a sleek (sometimes scifi) display case with transparent walls, and a minimalistic background.
    - Suggested prompts:
      - \<concept\>, micro-details, photorealism, photorealistic, \<kc-vx-iter\>, photorealistic, \<concept\>, very detailed, scifi case, \<kc-vx-iter\>, \<concept\>, very detailed, scifi transparent case, \<kc-vx-iter\>
    - Suggested negative prompts:
       - blurry, toy, cartoon, animated, underwater, photoshop
    - Suggested samplers:
       - DPM++ SDE Karras (used for the example images) or DPM++ 2S a Karras


### Model Resources:

 - Tron Legacy Diffusion [Huggingface Hub](https://huggingface.co/dallinmackay/Tron-Legacy-diffusion)
 - Robo Diffusion [Huggingface Hub](https://huggingface.co/nousr/robo-diffusion)
 - Classic Anim Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/classic-anim-diffusion)
 - Archer Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/archer-diffusion)
 - Spider-verse Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/spider-verse-diffusion)
 - Elden Ring Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/elden-ring-diffusion)
 - Mo(dern) Di(sney) Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/mo-di-diffusion)
 - Arcane Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/Arcane-Diffusion)
 - Pokemon Diffusion [Huggingface Hub](https://huggingface.co/lambdalabs/sd-pokemon-diffusers)
 - Waifu Diffusion [Huggingface Hub](https://huggingface.co/hakurei/waifu-diffusion)
 - Cyberpunk Anime Diffusion [Huggingface Hub](https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion)
 - Redshift Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/redshift-diffusion)
 - Ghibli Diffusion [Huggingface Hub](https://huggingface.co/nitrosocke/Ghibli-Diffusion)
 - Open Journey (aka Midjourney v4) Diffusion [Huggingface Hub](https://huggingface.co/prompthero/openjourney)
 - Knollingcase Diffusion [Huggingface Hub](https://huggingface.co/Aybeeceedee/knollingcase)
 - Anything v3 Diffusion [Huggingface Hub](https://huggingface.co/Linaqruf/anything-v3.0)
 - Inkpunk Diffusion [Huggingface Hub](https://huggingface.co/Envvi/Inkpunk-Diffusion)
 - Isopixel Diffusion [Huggingface Hub](https://huggingface.co/nerijs/isopixel-diffusion-v1)
 - Robo Diffusion 2 [Huggingface Hub](https://huggingface.co/nousr/robo-diffusion-2-base)
 - Dreamlike Diffusion 1.0 [Huggingface Hub](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0)
 - Waifu Diffusion v1.4 [Huggingface Hub](https://huggingface.co/hakurei/waifu-diffusion-v1-4)


### Embedding Resources:
 - Action Helper [Huggingface Hub](https://huggingface.co/spaablauw/ActionHelper)
 - Cinema Helper [Huggingface Hub](https://huggingface.co/spaablauw/CinemaHelper)
 - Photo Helper [Huggingface Hub](https://huggingface.co/spaablauw/PhotoHelper)
 - Vintage Helper [Huggingface Hub](https://huggingface.co/spaablauw/VintageHelper)
 - Knollingcase Embeddings for Stable Diffusion v2.0 [Huggingface Hub](https://huggingface.co/ProGamerGov/knollingcase-embeddings-sd-v2-0)