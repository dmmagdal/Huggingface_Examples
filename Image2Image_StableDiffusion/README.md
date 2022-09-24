# Image to Image with Stable Diffusion

Description: Leverage the image to image text guided generation with stable diffusion model for generating art from text prompts.


### Requirements

 - The version of the diffusers library outlined in `requirements.txt` requires Python 3.7+.
 - Create a `.env` file containing your Huggingface Hub user access token. This token is required to download the stable diffusion model. You must also agree to the huggingface terms of service associated with downloading the model. If there is a permission error the first time running this, visit the help link provided in the message.


### Notes:

 - A docker image is able to be created with the `Dockerfile`. The docker image created will run on CPU only (Docker connection to GPU needs to be explored).
 - The default revision/dtype of the model is float32. Use `revision="fp16"` and `torch_dtype=torch.float16` for running on GPUs with less than 10GB VRAM.
 - For running the code on Google Colab, add the following:
 	- Enable GPU access.
 	- `from huggingface_hub import notebook_login` will allow you to log into huggingface through the notebook when `notebook_login()`  is called/run.
 	- Enable external widgets:
 		```
 		from google.colab import output
 		output.enable_custom_widget_manager()
 		```
 - Run inference with Pytorch's autocast module when using GPU/"cuda". There is some variability to be expected in results, however there are also a number of parameters that can be tweaked such as `guidance_scale`, `number_of_steps`, and setting random seed (for deterministic results) that should help get more consistent results.
 - Differences between running on CPU vs GPU:
    - Do not specify the `revision` or `torch_dtype` arguments in `from_pretrained()` when running on CPU.
    - The amount of RAM used when running on CPU can be between 6 to 12GB. The amount of VRAM used when running on GPU is around 10 GB.
    - The amount of time it takes to run the model to produce 1 sample (`num_inference_steps=50` by default) on CPU is around 25 to 30 minutes while the same run would take around 12 to 25 seconds on GPU.
    - NOTE: I was unable to run the model on GPU on my GeForce RTX 2060 SUPER (6GB) on my Dell desktop. GPU measurements come from running the code on Google Colab (free tier, GPU - Tesla T4 16GB).
 - There is no need to worry about the difference between adding `revision="fp16"` and `torch_dtype=torch.float16` to the `from_pretrained()` method when downloading the model. Regardless of whether those arguments are specified, the full model is downloaded with no difference in the weights. Only the when the model is loaded are the weights "changed" or converted to the corresponding revision/dtype.
    - Again, default revision/dtype is float32.
    - The conversion to fp16 is for (consumer) GPUs which don't have as much VRAM as regular RAM.
 - On Windows, you will have to either run the program as administrator or enable developer mode. I would recommend just running the program as administrator for more better overall security than enabling developer mode. Enable developer mode at your own risk. For more information, see the following [Microsoft page](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).
 - Pipeline call arguments ([source](https://github.com/huggingface/diffusers/blob/ab7a78e8f11eec914653e01ee497d57d7503bd9d/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)):
   - `prompt`: str or List[str], the prompt(s) to guide the image generator.
   - `strength`: torch.FloatTensor or PIL.Image.Image, image or tensor representing an image batch, that wil be used as the starting point for the process.
   - `strength` Optional[float], conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1. `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`. Default is 0.0.
   - `num_inference_steps`: Optional[int], the number of denoising steps. More denoising steps usually lead to higher quality image at the expense of slower inference. Default is 50.
   - `guidance_scale`: Optional[float], guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality. Default is 7.5.


### Resources:

 - Huggingface Diffusers [GitHub repo](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)
 - Huggingface hub [integration page](https://huggingface.co/docs/hub/models-adding-libraries)
 - [Huggingface blog](https://huggingface.co/blog/stable_diffusion) on Stable Diffusion