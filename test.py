from typing import List,Union
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pyrallis
from util.masactrl import MutualSelfAttentionControl
from util.model import DINOHead,MLP

# from safetensors import safe_open
from diffusers import (
    # AutoencoderKL,
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline
    # UNet2DConditionModel,
)
# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch
from config import RunConfig
from util import ptp_utils
from util.pipeline_attend_and_excite import (AttendAndExcitePipeline, BlendedDiffusionPipeline,
                                             TextaulStableDiffusionPipeline,
                                             TextaulStableDiffusionXLPipeline,
                                             TextaulandContrPipeline)
from util.ptp_utils import AttentionBase, AttentionStore,TextualControl,MasaAttentionStore, regiter_attention_editor_diffusers
from diffusers import DPMSolverMultistepScheduler

def run_on_prompt(prompt:Union[str, List[str]],
                  model: Union[TextaulStableDiffusionPipeline,
                               AttendAndExcitePipeline,
                               TextaulandContrPipeline,
                               StableDiffusionPipeline,
                               TextaulStableDiffusionXLPipeline,
                               StableDiffusionXLPipeline],
                  controller: Union[AttentionStore, TextualControl],
                  indices_to_alter: List[int],
                  seed: torch.Generator,
                  config: RunConfig,
                  referenceImages: List[torch.Tensor] = None,
                  referenceLocattion: List[int] = None,
                  referenceText: str = None,
                  pixel_values: torch.Tensor =None) -> Image.Image:
    if controller is not None:
        if isinstance(controller, AttentionStore):
            ptp_utils.register_attention_control(model, controller)
        else:
            pass
            # ptp_utils.register_textual_attention_control(model, controller)
    if isinstance(model,AttendAndExcitePipeline):
        outputs = model(prompt=prompt,
                        attention_store=controller,
                        indices_to_alter=indices_to_alter,
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
        image = outputs.images[0]
    elif isinstance(model,TextaulandContrPipeline):
        ptp_utils.register_textual_attention_control(model,controller,last_layer=16)
        outputs = model(prompt=prompt,
                        referenceImages = referenceImages,
                        referenceLocattion = referenceLocattion,
                        controller=controller,
                        guidance_scale=config.guidance_scale,
                        num_inference_steps=config.n_inference_steps,
                        scale_range=config.scale_range,
                        generator=seed,
                        pixel_values=pixel_values,
                        )
        image = outputs.images[0]
    elif isinstance(model,BlendedDiffusionPipeline):
        outputs = model(prompt=prompt,
                        referenceImages = referenceImages,
                        referenceLocattion = referenceLocattion,
                        attention_store=controller,
                        guidance_scale=config.guidance_scale,
                        num_inference_steps=config.n_inference_steps,
                        scale_range=config.scale_range,
                        generator=seed,
                        pixel_values=pixel_values,
                        )
        image = outputs.images[0]
    elif isinstance(model,TextaulStableDiffusionPipeline):
        outputs = model(prompt=prompt,
                        referenceImages = referenceImages,
                        referenceLocattion = referenceLocattion,
                        attention_store=controller,
                        num_inference_steps = config.n_inference_steps,
                        guidance_scale = config.guidance_scale,
                        generator=seed,
                        )
        image = outputs.images[0]
    elif isinstance(model,TextaulStableDiffusionXLPipeline):
        prompt_2 = prompt
        outputs = model(prompt=prompt,
                        prompt_2=prompt_2,
                        referenceImages = referenceImages,
                        referenceLocattion = referenceLocattion,
                        num_inference_steps = config.n_inference_steps,
                        guidance_scale = config.guidance_scale,
                        generator=seed,
                        height=768,
                        width=768,
                        )
        image = outputs.images[0]
    elif isinstance(model,StableDiffusionXLPipeline):
        start_code = torch.randn([1, 4, 128, 128], device=model._execution_device, dtype=model.dtype)
        start_code = start_code.expand(len(prompt), -1, -1, -1)
        STEP = 4
        LAYER= 4
        controller = MutualSelfAttentionControl(STEP, LAYER, model_type="SD")
        regiter_attention_editor_diffusers(model, controller)
        outputs = model(prompt=prompt,
                        latents=start_code,
                        num_inference_steps=config.n_inference_steps,
                        guidance_scale=config.guidance_scale,
                        )
        image = outputs.images[-1]
    else:
        prompts = [
        "A portrait of an old man, facing camera, best quality",
        "A portrait of an old man, facing camera, smiling, best quality",
        ]
        start_code = torch.randn([1, 4, 64, 64], device=model._execution_device, dtype=model.dtype)
        start_code = start_code.expand(len(prompt), -1, -1, -1)
        STEP = 4
        LAYER= 4
        controller = MutualSelfAttentionControl(STEP, LAYER, model_type="SD")
        regiter_attention_editor_diffusers(model, controller)
        outputs = model(prompt=prompts,
                        num_inference_steps = config.n_inference_steps,
                        guidance_scale = config.guidance_scale,
                        latents=start_code,
                        )
        image = outputs.images[0]
    return image

@pyrallis.wrap()
def main(config: RunConfig):
    # from transformers import CLIPTextModel, CLIPTokenizer,CLIPImageProcessor
    root = "/opt/data/private/stable_diffusion_model"
    textual_name = "textual_inversion_find_xl_mixed_768_1"
    # placeholder_token = '<style>'
    nums_token = 768
    #"runwayml/stable-diffusion-v1-5"
    #"CompVis/stable-diffusion-v1-4"
    #stabilityai/stable-diffusion-2-1-base
    #stabilityai/stable-diffusion-2-1
    #stabilityai/sd-turbo
    #stabilityai/stable-diffusion-xl-base-1.0
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    name =''
    for x in model_id.split('/'):
        name += x+'_'
    name =name+f"pca_{ 768 if nums_token==1024 else nums_token}.pt"
    save_path = os.path.join("token_dict", name)
    token_dict = torch.load(save_path)
    bottleneck_dim = token_dict['features'].shape[-1]

    moldel_root = f'{root}/{textual_name}/model.pt'
    state_dict = torch.load(moldel_root)['model']
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',map_location='cpu')
    # head = DINOHead(in_dim=768, out_dim=nums_token,cls_dim=6, nlayers=3)
    head = MLP(in_dim=768, out_dim=nums_token,bottleneck_dim=bottleneck_dim,nlayers=3)
    image_model = torch.nn.Sequential(backbone, head)
    image_model.load_state_dict(state_dict)
    pipe = TextaulStableDiffusionXLPipeline.from_pretrained(repo_id,
                                                    torch_dtype=torch.float16)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    setattr(pipe, 'image_model', image_model)
    setattr(pipe, 'token_dict', token_dict)
    device = torch.device('cuda:0')
    pipe.to(device)
    prompts = [
            # "Painting of a S and a cute S in the style of S",
            # 'a photo of dog and cat',
            'a photo of # and #',
            # 'a photo of a # #',
            # 'Painting of a S in the corner in the style of S',
        ]
    referenceText = None
    image_size = 224
    crop_pct = 0.875
    interpolation = 3
    trans = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct),interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),])
    image_path =[
        # "datasets/Mixed_dataset/panda/a bright photo of the panda_124.jpg",
        # "dataset_mixed/mask.png",
        # "datasets/Mixed_dataset/teddybear/marina-shatskih-kBo2MFJz2QU-unsplash.jpg",
        # "datasets/Mixed_dataset/cat/jeanie-de-klerk-av2WGfogjqg-unsplash.jpg",
        # "datasets/Mixed_dataset/Gta5/R.jpg",
        # "datasets/Mixed_dataset/Pot/0.png",
        # "datasets/Mixed_dataset/Ande/Q.jpg",
        "datasets/Mixed_dataset/panda/a bright photo of the panda_202.jpg",
        # "datasets/Mixed_dataset/Fauvism/albert-marquet_street-lamp-arcueil-1899.jpg",
        ]
    referenceImage = []
    image_name = ''
    pixel_values = None
    # mask = torch.zeros([768, 768])
    # mask[568:668, 568:668] = 1
    for i,path in enumerate(image_path):
        image = Image.open(path)
        image = image.convert("RGB")
        if i==1:
            img = image
            img = np.array(img).astype(np.uint8)
            img = Image.fromarray(img)
            img = img.resize((512, 512), resample=Image.BILINEAR)
            img= np.array(img).astype(np.uint8)
            img = (img / 127.5 - 1.0).astype(np.float32)
            pixel_value = torch.from_numpy(img).permute(2, 0, 1)
            pixel_value = pixel_value.unsqueeze(0)
            pixel_values = pixel_value
            # continue
        image_name +=(path.split('/')[-2].split('.')[0]+'_')
        referenceImage.append(trans(image))
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    token_indices = [3,5]
    indices_to_alter = [5]
    # referenceImage =[]
    # token_indices = []
    controller = AttentionStore()
    # controller = TextualControl()
    # controller = None
    for prompt in prompts:
        os.makedirs(f"./images/{textual_name}/{prompt}",exist_ok=True)
        for seed in range(0,5):
            g = torch.Generator('cuda').manual_seed(seed)
            image = run_on_prompt(prompt=prompt,
                                model=pipe,
                                controller=controller,
                                indices_to_alter = indices_to_alter,
                                seed=g,
                                config=config,
                                referenceImages = referenceImage,
                                referenceLocattion = token_indices,
                                referenceText = referenceText,
                                pixel_values=pixel_values)
            image.save(f"./images/{textual_name}/{prompt}/{image_name}{seed}.png")

if __name__ == '__main__':
    main()