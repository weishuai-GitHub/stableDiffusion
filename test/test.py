from typing import List,Union
import torch
from torchvision import transforms
from PIL import Image
import os
import pyrallis
from  util.model import DINOHead

from safetensors import safe_open
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    # StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch
from config import RunConfig
from util import ptp_utils
from util.pipeline_attend_and_excite import AttendAndExcitePipeline
from util.ptp_utils import AttentionStore
from util.pipeline_attend_and_excite import TextaulStableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer,CLIPImageProcessor
root = "/opt/data/private/stable_diffusion_model"
textual_name = "textual_inversion_find_new_style_11"
placeholder_token = '<style>'
nums_token = 10
#"runwayml/stable-diffusion-v1-5"
#stabilityai/stable-diffusion-2-1-base
#stabilityai/stable-diffusion-2-1
#stabilityai/stable-diffusion-xl-base-1.0
model_id = "stabilityai/stable-diffusion-2-1"
moldel_root = f'model/{textual_name}_10000.pt'
state_dict = torch.load(moldel_root)['model']
backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',map_location='cpu')
head = DINOHead(in_dim=768, out_dim=nums_token, nlayers=3)
image_model = torch.nn.Sequential(backbone, head)
image_model.load_state_dict(state_dict)
# path = os.path.join(root,textual_name,"learned_embeds.safetensors")
path = os.path.join(root,textual_name,"learned_embeds-steps-30000.safetensors")

weight = {}
with safe_open(path, framework="pt", device=0) as f:
    for k in f.keys():
        weight[k] = f.get_tensor(k)
additional_tokens = []

# model_bin = os.path.join(root,textual_name,"checkpoint-14000/pytorch_model.bin")
# optimizer = torch.load(model_bin)
for i in range(0, nums_token):
    additional_tokens.append(f"{placeholder_token}_{i}")
placeholder_tokens = additional_tokens

# Load scheduler and models
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(model_id,subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder",
)
num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
# Resize the token embeddings as we are adding new special tokens to the tokenizer
text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
with torch.no_grad():
    for i,token_id in enumerate(placeholder_token_ids):
        token_embeds[token_id] = weight[placeholder_token][i]
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae",)
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", 
)
feature_extractor = CLIPImageProcessor.from_pretrained(model_id,subfolder="feature_extractor")
safety_checker = None
if os.path.exists(os.path.join(model_id,"safety_checker")):
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id,subfolder="safety_checker")

pipe = TextaulStableDiffusionPipeline(vae=vae, unet=unet, 
                               text_encoder=text_encoder,
                               tokenizer=tokenizer, 
                               scheduler=noise_scheduler,
                               feature_extractor=feature_extractor,
                               safety_checker=safety_checker,
                               image_model = image_model,
                               placeholder_token_ids = placeholder_token_ids,
                               )
pipe = pipe.to("cuda")

def run_on_prompt(prompt:Union[str, List[str]],
                  model: TextaulStableDiffusionPipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig,
                  referenceImages: List[torch.Tensor] = None,
                  referenceLocattion: List[int] = None,) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    if isinstance(model,AttendAndExcitePipeline):
        outputs = model(prompt=prompt,
                        attention_store=controller,
                        indices_to_alter=token_indices,
                        attention_res=config.attention_res,
                        guidance_scale=config.guidance_scale,
                        generator=seed,
                        num_inference_steps=config.n_inference_steps,
                        max_iter_to_alter=config.max_iter_to_alter,
                        run_standard_sd=config.run_standard_sd,
                        thresholds=config.thresholds,
                        scale_factor=config.scale_factor,
                        scale_range=config.scale_range,
                        smooth_attentions=config.smooth_attentions,
                        sigma=config.sigma,
                        kernel_size=config.kernel_size,
                        sd_2_1=config.sd_2_1)
    else:
        outputs = model(prompt=prompt,
                        referenceImages = referenceImages,
                        referenceLocattion = referenceLocattion,
                        num_inference_steps = config.n_inference_steps,
                        guidance_scale = config.guidance_scale,
                        generator=seed)
    image = outputs.images[0]
    return image

@pyrallis.wrap()
def main(config: RunConfig):
    prompts = [
            "a photo of in the **",
            # "a photo of the <style>_1,4k",
            # "a photo of the <style>_2,4k",
            # "a photo of the <style>_3,4k",
            # "a photo of the <style>_4,4k",
            # "a photo of the <style>_5,4k",
            # "a photo of the <style>_6,4k",
            # "a photo of the <style>_7,4k",
            # "a photo of the <style>_8,4k",
            # "a photo of the <style>_9,4k",
            # "a photo of the <object>_9 is on the grass",
            ] 
    # objiect_name = ["panda","cat","dog","anemone fish","hen","bee eater","box turtle","African elephant","rat","lion"]
    # for name in objiect_name:
    #     prompts.append(f"A {name} is drinking water,4k")   
    image_size = 224
    crop_pct = 0.875
    interpolation = 3
    trans = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct),interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),])
    referenceImage = Image.open('datasets/sub_style_dataset/Baroque/adriaen-brouwer_the-schlachtfest.jpg')
    referenceImage = trans(referenceImage)
    referenceImage = [referenceImage]
    token_indices = [5]
    # referenceImage =[]
    # token_indices = []
    controller = AttentionStore()
    for prompt in prompts:
        os.makedirs(f"./images/{textual_name}/{prompt}",exist_ok=True)
        for seed in range(0,10):
            g = torch.Generator('cuda').manual_seed(seed)
            image = run_on_prompt(prompt=prompt,
                                model=pipe,
                                controller=controller,
                                token_indices=token_indices,
                                seed=g,
                                config=config,
                                referenceImages = referenceImage,
                                referenceLocattion = token_indices)
            image.save(f"./images/{textual_name}/{prompt}/{seed}.png")

if __name__ == '__main__':
    main()