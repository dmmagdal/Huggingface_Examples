# Text to Image with Miscellaneous Stable Diffusion Models

Description: Leverage various fine-tuned diffusion models for generating art from text prompts.


### Requirements

 - The version of the diffusers library outlined in `requirements.txt` requires Python 3.7+.
 - The `ipywidgets>=7,<8` library is only required if running this code in a Google Colab environment (this allows for images to be displayed within the notebook).
 - Create a `.env` file containing your Huggingface Hub user access token. This token is required to download the stable diffusion model. You must also agree to the huggingface terms of service associated with downloading the model. If there is a permission error the first time running this, visit the help link provided in the message.


### Notes:

 - Refer to the following README.md Notes sections from previous Stable Diffusion examples for guidance on running Stable Diffusion (including text to image, image to image, and image inpainting): [Text2Image_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/WaifuDiffusion_StableDiffusion/README.md), [Image2Image_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/Image2Image_StableDiffusion/README.md), and [InPainting_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/InPainting_StableDiffusion/README.md) [WaifuDiffusion_StableDiffusion](https://github.com/dmmagdal/Huggingface_Examples/blob/main/WaifuDiffusion_StableDiffusion/README.md)
 - This example primarily deals with just text to image. However the [Finetuned Diffusion](https://huggingface.co/spaces/anzorq/finetuned_diffusion) model/demo shows that these models can be applied to image to image and inpainting applications.
 - **Stable Diffusion (v1.4) Official V1 Release**
    - Checkpoint resumed from stable-diffusion-v1-2. 195,000 steps at 512x512 resolution on "laion-improved-aesthetics" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling
    - The "original" official v1 release of Stable Diffusion from Stability AI through CompVis
 - **Stable Diffusion (v1.5) Unofficial Release**
    - Checkpoint resumed from stable-diffusion-v1-2. Finetuned on 595,000 steps at 512x512 resolution on "laion-improved-aesthetics v2.5+" and 10% dropping of the text-conditioning to improve classifier-free guidance sampling
    - The unofficial v1.5 release of Stable Diffusion from RunwayML
 - **Stable Diffusion (v2.0) Official V2 Release**
    - Resumed training from stable-diffusion-2-base (512-base-ema-ckpt) and trained for 150,000 steps using a v-objective on the same dataset. Resumed for another 140,000 steps on 768x768 images
    - The unofficial v1.5 release of Stable Diffusion from RunwayML
 - Running Stable Diffusion v2 requires an updated Huggingface Transformers and Diffusers modules as well as Huggingface's Accelerate module. Stable Diffusion v2 also will OOM on VRAM for consumer GPUs (such as my Nvidia 2060 SUPER at 8GB VRAM), so an additional step of adding this line `pipe.enable_attention_slicing()` is required after calling `pipe = pipe.to('cuda')` to send the pipeline instance to the GPU. This will reduce VRAM consumption at the cost of speed. NOTE: Running 1 image inference (batch size 1) on my GPU uses almost all of the VRAM. Hopefully Stablility AI or the community will come out with a more lightweight model or I'll just use v1 branches of Stable Diffusion
    - Another caveat to the updated diffusers packages (previously I had been using `diffusers==0.3.0` and `transformers==4.17.0` for stable diffusion v1+) is that the output of the pipeline is not a dictionary anymore. Rather than call `image = pipe(prompt)["sample"][0]` to get the image, use `image = pipe(prompt).images[0]` for this version of the modules (`diffusers==0.9.0` and `transformers==4.24.0`)


### Resources:

 - Stable Diffusion (v1.4) [Huggingface Hub](https://huggingface.co/CompVis/stable-diffusion-v1-4)
 - Stable Diffusion v1.5 [Huggingface Hub](https://huggingface.co/runwayml/stable-diffusion-v1-5)
 - Stable Diffusion v2 [Huggingface Hub](https://huggingface.co/stabilityai/stable-diffusion-2)
 - Stable Diffusion v2.1 [Huggingface Hub](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)