import torch
from diffusers import StableDiffusionXLPipeline,AutoPipelineForText2Image
model_id = '../stabilityai/stable-diffusion-2-1'
pipeline = AutoPipelineForText2Image.from_pretrained(model_id,torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to('cuda:0')

import os
import random
import matplotlib.pyplot as plt
prompts = ['pot']
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a dark photo of the {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]
root = "../dataset_mixed/"
for p in prompts:
    image_path = os.path.join(root,p)
    os.makedirs(image_path,exist_ok=True)
    for seed in range(0,100):
        templates = random.choice(imagenet_templates_small)
        prompt = templates.format(p)
        g = torch.Generator('cuda').manual_seed(seed)
        output = pipeline(prompt,generator=g)
        image = output.images[0]
        image.save(f"{image_path}/{prompt}_{seed}.jpg")
print("Done!")