import torch
from diffusers import StableDiffusionXLPipeline,StableDiffusionPipeline
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
model_id = 'runwayml/stable-diffusion-v1-5'
pipeline = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to('cuda:1')

seed = 42
seed_everything(seed)
device = pipeline.device
# prompts = ["A portrait of an old man, facing camera, best quality",
#           "A portrait of an old man, facing camera, smiling, best quality"
# ]
prompts = "A portrait of an old man, facing camera, best quality"
prompts_2 = "A portrait of an old man, facing camera, smiling, best quality"
start_code = torch.randn([1, 4, 64, 64], device=device,dtype=torch.float16)
# start_code = start_code.expand(len(prompts), -1, -1, -1)
output = pipeline(prompts,prompt_2 = prompts_2, latents=start_code, guidance_scale=7.5)
image1 = output.images[0]
image1.save(f"A portrait of an old man, facing camera, best quality.jpg")
# image2 = output.images[1]
# image2.save(f"A portrait of an old man, facing camera, smiling, best quality.jpg")
# plt.subplot(1,2,1)
# plt.imshow(image1)
# plt.subplot(1,2,2)
plt.imshow(image1)