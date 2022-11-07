# Text to Image with Miscellaneous Stable Diffusion Models

Description: Leverage various fine-tuned diffusion models for generating art from text prompts.


### Requirements

 - The version of the diffusers library outlined in `requirements.txt` requires Python 3.7+.
 - The `ipywidgets>=7,<8` library is only required if running this code in a Google Colab environment (this allows for images to be displayed within the notebook).
 - Create a `.env` file containing your Huggingface Hub user access token. This token is required to download the stable diffusion model. You must also agree to the huggingface terms of service associated with downloading the model. If there is a permission error the first time running this, visit the help link provided in the message.


### Notes:

 - Refer to the following README.md Notes sections from previous Stable Diffusion examples for guidance on running Stable Diffusion (including text to image, image to image, and image inpainting): [Text2Image_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/WaifuDiffusion_StableDiffusion/README.md), [Image2Image_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/Image2Image_StableDiffusion/README.md), and [InPainting_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/InPainting_StableDiffusion/README.md) [WaifuDiffusion_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/WaifuDiffusion_StableDiffusion/README.md)
 - This example primarily deals with just text to image. However the [Finetuned Diffusion](https://huggingface.co/spaces/anzorq/finetuned_diffusion) model/demo shows that these models can be applied to image to image and inpainting applications.
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


### Resources:

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