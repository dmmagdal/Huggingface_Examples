# Huggingface Examples

Description: Some example programs that use Huggingface datasets, transformers, and tokenizer libraries.


### Fine-tuning GPT-2 for Magic the Gathering
Status: Complete (Validated)


### Fine-tuning BERT on IMDB
Status: Complete (Validating)


### Image2Image with Stable Diffusion
Status: Complete (Validated)


### InPainting with Stable Diffusion
Status: Complete (Validated)


### Huggingface with Tensorflow 2.0
Status: Complete (Validated)


### Text2image with Stable Diffusion
Status: Complete (Validated)


### Additional Sources:
 - [Huggingface Transformers course](https://huggingface.co/course/chapter1/1)

### Notes:
 - If you can't get an example to run on docker, chances are I ran it on my Dell desktop bare metal (Intel i7-10700 CPU @ 2.90GHz, RAM 16GB, GPU Nvidia GeForce RTX 2060 SUPER 8GB VRAM). There is a slight bit of resource overhead when running an example on Docker in Windows 10 vs running the program bare metal on the OS.
 - I dont want to keep updating the `Text2Image_StableDiffusion/` or the `MiscStyles_StableDiffusion/` folders when there are stable diffusion updates. Please refer to the following below for any additional major updates with Stable Diffusion.
	 - Stable Diffusion XL
		 - [Pipeline documentation](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
		 - Stable Diffusion XL 1.0 (base) [model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
		 - Stable Diffusion XL 1.0 (refiner) [model card](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)
		 - Stable Diffusion XL is very large (and as a result, resource hungry). If it fails to initialize on a device like the DarkStar GPU server, make sure the primary GPU card is available to run it (see this relevant [GitHub issue thread](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/11685)).