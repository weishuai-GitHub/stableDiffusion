import os
from diffusers import (DDIMScheduler,
                       AutoencoderKL,
                       UNet2DConditionModel,
                       StableDiffusionPipeline)
from util.attention_based_segmentation import get_cluter,cluster2noun,get_background_mask
from util.ptp_utils import AttentionStore,register_attention_control
from PIL import Image
import numpy as np
import torch
device = 'cuda'
model_id = 'CompVis/stable-diffusion-v1-4'
model = StableDiffusionPipeline.from_pretrained(model_id)
model = model.to(device)
vae = model.vae
scheduler = model.scheduler
unet = model.unet

prompt = 'a photo of panda'
text_inputs = model.tokenizer(
                prompt,
                padding="max_length",
                max_length=model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
text_input_ids = text_inputs.input_ids
prompt_embeds = model.text_encoder(text_input_ids.to(device))[0]

def get_latent(image,t,noises):
    image = image.to('cuda')
    latent = vae.encode(image).latent_dist.mean
    noises_latent = scheduler.add_noise(latent, noises, t)
    return noises_latent

image_path = 'datasets/sample_dataset/panda/a good photo of a panda_613.jpg'

image = Image.open(image_path)
image = image.convert("RGB")

img = image
img = np.array(img).astype(np.uint8)
img = Image.fromarray(img)
img = img.resize((512, 512), resample=Image.BILINEAR)
img= np.array(img).astype(np.uint8)
img = (img / 127.5 - 1.0).astype(np.float32)
pixel_value = torch.from_numpy(img).permute(2, 0, 1)
pixel_value = pixel_value.unsqueeze(0)
name = image_path.split('/')[-2]
scheduler.set_timesteps(50, device='cuda')
controller = AttentionStore()
register_attention_control(model,controller)
noises = torch.randn((1, 4,64,64)).to('cuda')
# latent = noises
with torch.no_grad():
    os.makedirs(f'latent/{name}',exist_ok=True)
    for i,t in enumerate(scheduler.timesteps):
        latent = get_latent(pixel_value,t,noises)
        latent_model_input = scheduler.scale_model_input(latent, t)
        noise = unet(latent_model_input,t,encoder_hidden_states=prompt_embeds)
        mask = get_cluter(controller,num_segments=5)
        segment_mask = Image.fromarray((255*mask/5).astype(np.uint8))
        token_map = cluster2noun(controller,mask,[i for i in range(len(prompt.split(' ')))],num_segments=5)
        background_mask = get_background_mask(mask,token_map,num_segments=5)
        min_v = background_mask.min()
        max_v = background_mask.max()
        background_mask = (background_mask - min_v) / (max_v - min_v)
        background_mask = 255*background_mask
        background_mask = background_mask.astype(np.uint8)
        background_mask = Image.fromarray(background_mask)
        background_mask.save(f'latent/{name}/background_{t}.png')
        segment_mask.save(f'latent/{name}/segment_{t}.png')
        
