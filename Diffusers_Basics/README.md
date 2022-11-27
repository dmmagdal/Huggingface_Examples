# Getting Started with Huggingface Diffusers 

Description: A simple resource to look into getting started with using Huggingface's Diffusers library for different applications.


### Notes:

 - The Huggingface Diffusers library offers several pipelines, noise schedulers, different models, and training examples.
 - The following table shows the officially supported pipelines and corresponding papers:

 | Pipeline 				| Paper 																		| Tasks 									|
 | ------------------------ | ----------------------------------------------------------------------------- | ----------------------------------------- |
 | dance_diffusion 			| Dance Diffusion 																| Unconditional Audio Generation 			|
 | ddpm 					| Denoising Diffusion Probabilistic Models 										| Unconditional Image Generation 			|
 | ddim 					| Denoising Diffusion Implicit Models 											| Unconditional Image Generation 			|
 | latent_diffusion 		| High-Resolution Image Synthesis with Latent Diffusion Models 					| Text-to-Image Generation 					|
 | latent_diffusion_uncond 	| High-Resolution Image Synthesis with Latent Diffusion Models 					| Unconditional Image Generation 			|
 | pndm 					| Pseudo Numerical Methods for Diffusion Models on Manifolds 					| Unconditional Image Generation 			|
 | score_sde_ve 			| Score-Based Generative Modeling through Stochastic Differential Equations 	| Unconditional Image Generation 			|
 | score_sde_vq 			| Score-Based Generative Modeling through Stochastic Differential Equations 	| Unconditional Image Generation 			|
 | stable_diffusion 		| Stable Diffusion 																| Text-to-Image Generation 					|
 | stable_diffusion 		| Stable Diffusion 																| Image-to-Image Text-Guided Generation 	|
 | stable_diffusion 		| Stable Diffusion 																| Text-Guided Image Inpainting 				|
 | stochastic_karras_ve 	| Elucidating the Design Space of Diffusion-Based Generative Models 			| Unconditional Image Generation 			|
 | vq_diffusion 			| Vector Quantized Diffusion Model for Text-to-Image Synthesis 					| Text-to-Image Generation 					|


### Resources:

 - Huggingface Diffusers Module [Homepage](https://huggingface.co/docs/diffusers/index)
     - Huggingface Diffusers Module API (Main Classes):
    	 - [Model](https://huggingface.co/docs/diffusers/api/models)
    	 - [Schedulers](https://huggingface.co/docs/diffusers/api/schedulers)
    	 - [Diffusion Pipeline](https://huggingface.co/docs/diffusers/api/diffusion_pipeline)
    	 - [Logging](https://huggingface.co/docs/diffusers/api/logging)
    	 - [Configuration](https://huggingface.co/docs/diffusers/api/configuration)
    	 - [Outputs](https://huggingface.co/docs/diffusers/api/outputs)
     - Huggingface Diffusers Module API (Pipelines):
    	 - [Overview](https://huggingface.co/docs/diffusers/api/pipelines/overview)
    	 - [AltDiffusion](https://huggingface.co/docs/diffusers/api/pipelines/alt_diffusion)
    	 - [Cycle Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/cycle_diffusion)
    	 - [DDIM](https://huggingface.co/docs/diffusers/api/pipelines/ddim)
    	 - [DDPM](https://huggingface.co/docs/diffusers/api/pipelines/ddpm)
    	 - [Latent Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/latent_diffusion)
    	 - [Unconditional Latent Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/latent_diffusion_uncond)
    	 - [PNDM](https://huggingface.co/docs/diffusers/api/pipelines/pndm)
    	 - [Score SDE VE](https://huggingface.co/docs/diffusers/api/pipelines/score_sde_ve)
    	 - [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion)
    	 - [Stable Diffusion 2](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion_2)
    	 - [Safe Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion_safe)
    	 - [Stochastic Karras VE](https://huggingface.co/docs/diffusers/api/pipelines/stochastic_karras_ve)
    	 - [Dance Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/dance_diffusion)
    	 - [Versatile Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/versatile_diffusion)
    	 - [VQ Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/vq_diffusion)
    	 - [RePaint](https://huggingface.co/docs/diffusers/api/pipelines/repaint)
     - Huggingface Diffusers Training Examples:
     	 - [Unconditional Image Generation](https://huggingface.co/docs/diffusers/training/unconditional_training) \[[Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)\]
     	 - [Text-to-Image Fine-tuning](https://huggingface.co/docs/diffusers/training/text2image)
     	 - [Textual Inversion](https://huggingface.co/docs/diffusers/training/text_inversion) \[[Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb)\]
     	 - [Dreambooth](https://huggingface.co/docs/diffusers/training/dreambooth) \[[Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)\]