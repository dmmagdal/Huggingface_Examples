# Text to Image with Stable Diffusion

Description: Leverage the text to image model stable diffusion model for generating art from text prompts.


### Requirements

 - The version of the diffusers library outlined in `requirements.txt` requires Python 3.7+.
 - The `ipywidgets>=7,<8` library is only required if running this code in a Google Colab environment (this allows for images to be displayed within the notebook).
 - Create a `.env` file containing your Huggingface Hub user access token. This token is required to download the stable diffusion model. You must also agree to the huggingface terms of service associated with downloading the model. If there is a permission error the first time running this, visit the help link provided in the message.

### Notes:

 - A docker image is able to be created with the `Dockerfile` however there still needs to be some work done on how to run that image on CPU (currently only can run the model on GPU/"cuda").
 - The default revision/dtype of the model is float32. Use `revision="fp16"` and `torch_dtype=torch.float16` for running on GPUs with less than 10GB VRAM.
 - For running the code on Google Colab, add the following:
 	- Enable GPU access.
 	- `from huggingface_hub import notebook_login` will allow you to log into huggingface through the notebook when `notebook_login()`  is called/run.
 	- Enable external widgets:
 		```
 		from google.colab import output
 		output.enable_custom_widget_manager()
 		```
 - Run inference with Pytorch's autocast module. There is some variability to be expected in results, however there are also a number of parameters that can be tweaked such as `guidance_scale`, `number_of_steps`, and setting random seed (for deterministic results) that should help get more consistent results.


### Resources:

 - [Medium article](https://towardsdatascience.com/how-to-generate-images-from-text-with-stable-diffusion-models-ea9d1cb92f9b)
 - [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)
 - Model card for Stable Diffusion (v1.4) on [Huggingface hub](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 - Another [GitHub example](https://github.com/nicknochnack/StableDiffusionApp) 
 - [YouTube video](https://www.youtube.com/watch?v=7xc0Fs3fpCg) linked to the above GitHub example
 - Huggingface Diffusers [GitHub repo](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)
 - Huggingface hub [integration page](https://huggingface.co/docs/hub/models-adding-libraries)