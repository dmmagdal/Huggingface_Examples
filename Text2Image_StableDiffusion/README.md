# Text to Image with Stable Diffusion

Description: Leverage the text to image model stable diffusion model for generating art from text prompts.


### Requirements

 - The version of the diffusers library outlined in `requirements.txt` requires Python 3.7+.
 - The `ipywidgets>=7,<8` library is only required if running this code in a Google Colab environment (this allows for images to be displayed within the notebook).
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
    - The amount of RAM used when running on CPU can be between 6 to 12GB. The amount of VRAM used when running on GPU is around 4-10 GB.
    - The amount of time it takes to run the model to produce 1 sample (`num_inference_steps=50` by default) on CPU is around 6 to 9 minutes while the same run would take around 12 to 25 seconds on GPU.
 - There is no need to worry about the difference between adding `revision="fp16"` and `torch_dtype=torch.float16` to the `from_pretrained()` method when downloading the model. Regardless of whether those arguments are specified, the full model is downloaded with no difference in the weights. Only the when the model is loaded are the weights "changed" or converted to the corresponding revision/dtype.
    - Again, default revision/dtype is float32.
    - The conversion to fp16 is for (consumer) GPUs which don't have as much VRAM as regular RAM.
 - On Windows, you will have to either run the program as administrator or enable developer mode. I would recommend just running the program as administrator for more better overall security than enabling developer mode. Enable developer mode at your own risk. For more information, see the following [Microsoft page](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).


### Resources:

 - [Medium article](https://towardsdatascience.com/how-to-generate-images-from-text-with-stable-diffusion-models-ea9d1cb92f9b)
 - [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)
 - Model card for Stable Diffusion (v1.4) on [Huggingface hub](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 - Another [GitHub example](https://github.com/nicknochnack/StableDiffusionApp) 
 - [YouTube video](https://www.youtube.com/watch?v=7xc0Fs3fpCg) linked to the above GitHub example
 - Huggingface Diffusers [GitHub repo](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)
 - Huggingface hub [integration page](https://huggingface.co/docs/hub/models-adding-libraries)
 - [Huggingface blog](https://huggingface.co/blog/stable_diffusion) on Stable Diffusion