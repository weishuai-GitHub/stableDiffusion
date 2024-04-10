import torch
from PIL import Image
from diffusers import StableDiffusionPipeline,AutoPipelineForText2Image,StableDiffusionXLImg2ImgPipeline
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id,torch_dtype=torch.float16)
pipeline.unet.enable_xformers_memory_efficient_attention()
pipeline = pipeline.to('cuda:1')
image=Image.open('/opt/data/private/code/stableDiffusion/A portrait of an old man, facing camera, best quality.jpg')
prompt = "flamingos standing in the water near a tree."
g = torch.Generator('cuda').manual_seed(1)
output = pipeline(prompt=prompt,image=image,num_inference_steps=50,guidance_scale=7.5,generator=g)
image = output.images[0]
image.resize((512,512))
image.show()