from typing import List,Union
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pyrallis
import torchvision
from util.masactrl import MutualSelfAttentionControl
from util.model import DINOHead,MLP
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import gc
from torchvision.utils import save_image

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
from util.pipeline_attend_and_excite import (TextaulStableDiffusionXLPipeline,
                                             TextaulandContrXLPipeline)
from util.ptp_utils import AttentionBase, AttentionStore,TextualControl,MasaAttentionStore, regiter_attention_editor_diffusers
from diffusers import DPMSolverMultistepScheduler

def run_on_prompt(prompt:Union[str, List[str]],
                  model: Union[StableDiffusionXLPipeline,
                               TextaulStableDiffusionXLPipeline,
                               TextaulandContrXLPipeline],
                  controller: Union[AttentionStore, TextualControl],
                  seed: torch.Generator,
                  config: RunConfig,
                  referenceImages: List[torch.Tensor] = None,
                  referenceLocattion: List[int] = None,
                  is_pre: bool = True,
                  mask1:torch.Tensor = None,
                  mask2:torch.Tensor = None) -> Image.Image:
    torch.cuda.empty_cache()
    if controller is not None:
        if isinstance(controller, AttentionStore):
            ptp_utils.register_attention_control(model, controller)
        elif isinstance(controller, TextualControl):
            if isinstance(model, TextaulandContrXLPipeline):
                ptp_utils.register_textual_attention_control(model, controller,last_layer=8,model_type="SDXL")
    if isinstance(model,TextaulandContrXLPipeline):
        prompt_2 = prompt
        outputs = model(prompt=prompt,
                        prompt_2=prompt_2,
                        referenceImages = referenceImages,
                        referenceLocattion = referenceLocattion,
                        controller=controller,
                        num_inference_steps = config.n_inference_steps,
                        guidance_scale = config.guidance_scale,
                        generator=seed,
                        height=768,
                        width=768,
                        is_pre = is_pre,
                        seg_mask1 = mask1,
                        seg_mask2 = mask2
                        )
        image = outputs.images
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
        image = outputs.images        
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
        image = outputs.images
    return image

@pyrallis.wrap()
def main(config: RunConfig):
    # from transformers import CLIPTextModel, CLIPTokenizer,CLIPImageProcessor
    root = "/opt/data/private/stable_diffusion_model"
    # textual_name = "textual_inversion_find_xl_mixed_768"
    textual_name = "xl_mixed_768_4"
    # placeholder_token = '<style>'
    nums_token = 768
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    name =''
    for x in model_id.split('/'):
        name += x+'_'
    name =name+f"pca_{nums_token}.pt"
    save_path = os.path.join("token_dict", name)
    token_dict = torch.load(save_path)
    bottleneck_dim = token_dict['features'].shape[-1]
    bottleneck_dim_2 = token_dict['features_2'].shape[-1]

    moldel_root = f'{root}/{textual_name}/model.pt'
    _dict = torch.load(moldel_root)
    state_dict = _dict['model']
    state_dict_2 = _dict['model_2']
    #backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16',map_location='cpu')
    backbone = torch.hub.load('/opt/data/private/code/stableDiffusion/dino-main/', 'dino_vitb16', map_location='cpu', trust_repo=True, source='local')
    backbone_2 = torch.hub.load('/opt/data/private/code/stableDiffusion/dino-main/', 'dino_vitb16', map_location='cpu', trust_repo=True, source='local')
    # head = DINOHead(in_dim=768, out_dim=nums_token,cls_dim=6, nlayers=3)
    head = MLP(in_dim=768, out_dim=nums_token,bottleneck_dim=bottleneck_dim,nlayers=3)
    head_2 = MLP(in_dim=768, out_dim=960,bottleneck_dim=bottleneck_dim_2,nlayers=3)
    image_model = torch.nn.Sequential(backbone, head)
    image_model_2 = torch.nn.Sequential(backbone_2, head_2)
    image_model.load_state_dict(state_dict)
    image_model_2.load_state_dict(state_dict_2)

    
    pipe = TextaulandContrXLPipeline.from_pretrained(repo_id,
                                                    torch_dtype=torch.float16)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    device = torch.device('cuda:0')
    debug = False
    if not debug:
        mask_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        mask_model = mask_model.to(device)
        torch.set_printoptions(threshold=torch.inf)
        mask_model.eval()
    
    setattr(pipe, 'image_model', image_model)
    setattr(pipe, 'image_model_2', image_model_2)
    setattr(pipe, 'token_dict', token_dict)
    
    pipe.to(device)
    
    prompts = [
            # 'A dog in the style of #',
            [
                # 'A full-body portrait of a owl, standing close up, with the body filling most of the frame.',
                'A good photo of a # standing close up, with the body filling most of the frame.',
                'A full body portrait of a #.',
                # 'A good photo of a dog on the ground.',
                'a image image # and #.',
            ],
        ]
    image_path =[
        "datasets/animals/box turtle/a close-up photo of a box turtle_494.jpg",
        "datasets/animals/zucchini/1.png",
        # "datasets/new_datasets/rhinoceros/3c441456-72ae-4101-bb67-8f3a17370038.jpg",        
        # "datasets/animals/horse/horse11.png",
        # "datasets/animals_datasets/anemone fish/a photo of the clean anemone fish_494.jpg",
        # "datasets/animals_datasets/lion/panthera-leo_32_fe2ea055.jpg",
        ]
    
    token_indices = [
        [5],
        [6],
        [3,5],
    ]
    image_size = 224
    crop_pct = 0.875
    interpolation = 3
    trans = transforms.Compose([
        transforms.Resize(int(image_size / crop_pct),interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),])
    
    referenceImage0 = []
    referenceImage1 = []
    referenceImage2 = []
    referenceImage = []
    image_name = ''
    pixel_values = []
    is_pre = True
    # mask = torch.zeros([768, 768])
    # mask[568:668, 568:668] = 1
    for i,path in enumerate(image_path):
        image = Image.open(path)
        image = image.convert("RGB")
        img = image
        img = np.array(img).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize((512, 512), resample=Image.BILINEAR)
        img= np.array(img).astype(np.uint8)
        img = (img / 127.5 - 1.0).astype(np.float32)
        pixel_value = torch.from_numpy(img).permute(2, 0, 1)
        pixel_value = pixel_value.unsqueeze(0)
        pixel_values.append(pixel_value)
        image_name +=(path.split('/')[-2].split('.')[0]+'_')
        img = trans(image)
        if i == 0:
            referenceImage0.append(img)
            referenceImage2.append(img)
        if i == 1:
            referenceImage1.append(img)
            referenceImage2.append(img)
    assert len(referenceImage0) == len(token_indices[0]), "The number of reference images should be the same as the number of token_indices"
    assert len(referenceImage1) == len(token_indices[1]), "The number of reference images should be the same as the number of token_indices"
    referenceImage.append(referenceImage0)
    referenceImage.append(referenceImage1)

    # controller = AttentionStore()
    # controller = TextualControl()
    controller = TextualControl()
    for prompt in prompts:
        for i in range(len(prompt)):
            os.makedirs(f"./images/{textual_name}/{prompt[i]}",exist_ok=True)
        for seed in range(40,50):
            g = torch.Generator('cuda').manual_seed(seed) 
            is_pre = True
            images_pre = run_on_prompt(prompt=prompt,
                                model=pipe,
                                controller=controller,
                                seed=g,
                                config=config,
                                referenceImages = referenceImage,
                                referenceLocattion = token_indices,
                                is_pre = is_pre)
            
            for i,image in enumerate(images_pre):
                image.save(f"./images/{textual_name}/{prompt[i]}/{image_name}{i}_{seed}.png")
            is_pre = False
            binary_mask1,binary_mask2 = None,None
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            mask1_ = transform(images_pre[0]).to(device)
            mask2_ = transform(images_pre[1]).to(device)
            if not debug:
                mask1 = mask_model([mask1_])
                mask2 = mask_model([mask2_])

                # 设置阈值，将大于阈值的像素设为1，小于等于阈值的像素设为0
                threshold = 0.3  # 可根据实际情况调整阈值
                binary_mask1 = (mask1[0]['masks'][0, 0] > threshold).float()
                binary_mask2 = (mask2[0]['masks'][0, 0] > 0.8).float()
                g = torch.Generator('cuda').manual_seed(seed) 
                images = run_on_prompt(prompt=prompt,
                                    model=pipe,
                                    controller=controller,
                                    seed=g,
                                    config=config,
                                    referenceImages = referenceImage,
                                    referenceLocattion = token_indices,
                                    is_pre = is_pre,
                                    mask1 = binary_mask1,
                                    mask2 = binary_mask2)
                for i,image in enumerate(images):
                    image.save(f"./images/{textual_name}/{prompt[i]}/{image_name}new_{i}_{seed}.png")

if __name__ == '__main__':
    main()