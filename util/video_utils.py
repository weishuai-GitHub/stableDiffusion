import contextlib
import random
import numpy as np
import os
from glob import glob
from PIL import Image, ImageSequence

import torch
from torchvision.io import read_video, write_video
import torchvision.transforms as T

from einops import rearrange

FRAME_EXT = [".jpg", ".png"]



def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def join_frame(x, fsize):
    """ Join multi-frame tokens """
    x = rearrange(x, "(B F) N C -> B (F N) C", F=fsize)
    return x

def split_frame(x, fsize):
    """ Split multi-frame tokens """
    x = rearrange(x, "B (F N) C -> (B F) N C", F=fsize)
    return x

def func_warper(funcs):
    """ Warp a function sequence """
    def fn(x, **kwarg):
        for func in funcs:
            x = func(x, **kwarg)
        return x
    return fn

def join_warper(fsize):
    def fn(x):
        x = join_frame(x, fsize)
        return x
    return fn

def split_warper(fsize):
    def fn(x):
        x = split_frame(x, fsize)
        return x
    return fn

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = T.ToTensor()(image)
    return image.unsqueeze(0)


def process_frames(frames, h, w):

    fh, fw = frames.shape[-2:]
    h = int(np.floor(h / 64.0)) * 64
    w = int(np.floor(w / 64.0)) * 64

    nw = int(fw / fh * h)
    if nw >= w:
        size = (h, nw)
    else:
        size = (int(fh / fw * w), w)

    assert len(frames.shape) >= 3
    if len(frames.shape) == 3:
        frames = [frames]

    print(
        f"[INFO] frame size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

    frame_ls = []
    for frame in frames:
        resized_frame = T.Resize(size)(frame)
        cropped_frame = T.CenterCrop([h, w])(resized_frame)
        # croped_frame = T.FiveCrop([h, w])(resized_frame)[0]
        frame_ls.append(cropped_frame)
    return torch.stack(frame_ls)


def glob_frame_paths(video_path):
    frame_paths = []
    for ext in FRAME_EXT:
        frame_paths += glob(os.path.join(video_path, f"*{ext}"))
    frame_paths = sorted(frame_paths)
    return frame_paths


def load_video(video_path, h, w, frame_ids=None, device="cuda"):

    if ".mp4" in video_path:
        frames, _, _ = read_video(
            video_path, output_format="TCHW", pts_unit="sec")
        frames = frames / 255
    elif ".gif" in video_path:
        frames = Image.open(video_path)
        frame_ls = []
        for frame in ImageSequence.Iterator(frames):
            frame_ls += [T.ToTensor()(frame.convert("RGB"))]
        frames = torch.stack(frame_ls)
    else:
        frame_paths = glob_frame_paths(video_path)
        frame_ls = []
        for frame_path in frame_paths:
            frame = load_image(frame_path)
            frame_ls.append(frame)
        frames = torch.cat(frame_ls)
    if frame_ids is not None:
        frames = frames[frame_ids]

    print(f"[INFO] loaded video with {len(frames)} frames from: {video_path}")

    frames = process_frames(frames, h, w)
    return frames.to(device)


def save_video(frames: torch.Tensor, path, frame_ids=None, save_frame=False):
    os.makedirs(path, exist_ok=True)
    if frame_ids is None:
        frame_ids = [i for i in range(len(frames))]
    frames = frames[frame_ids]

    proc_frames = (rearrange(frames, "T C H W -> T H W C") * 255).to(torch.uint8).cpu()
    write_video(os.path.join(path, "output.mp4"), proc_frames, fps = 10, video_codec="h264")
    print(f"[INFO] save video to {os.path.join(path, 'output.mp4')}")

    if save_frame:
        save_frames(frames, os.path.join(path, "frames"), frame_ids = frame_ids)
    

def save_frames(frames: torch.Tensor, path, ext="png", frame_ids=None):
    os.makedirs(path, exist_ok=True)
    if frame_ids is None:
        frame_ids = [i for i in range(len(frames))]
    for i, frame in zip(frame_ids, frames):
        T.ToPILImage()(frame).save(
            os.path.join(path, '{:04}.{}'.format(i, ext)))


def load_latent(latent_path, t, frame_ids=None):
    latent_fname = f'noisy_latents_{t}.pt'

    lp = os.path.join(latent_path, latent_fname)
    assert os.path.exists(
        lp), f"Latent at timestep {t} not found in {latent_path}."

    latents = torch.load(lp)
    if frame_ids is not None:
        latents = latents[frame_ids]
    
    # print(f"[INFO] loaded initial latent from {lp}")

    return latents

@torch.no_grad()
def prepare_depth(pipe, frames, frame_ids, work_dir):
    
    depth_ls = []
    depth_dir = os.path.join(work_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    for frame, frame_id in zip(frames, frame_ids):
        depth_path = os.path.join(depth_dir, "{:04}.pt".format(frame_id))
        depth = load_depth(pipe, depth_path, frame)
        depth_ls += [depth]
    print(f"[INFO] loaded depth images from {depth_path}")
    return torch.cat(depth_ls)

# From pix2video: code/file_utils.py

def load_depth(model, depth_path, input_image, dtype=torch.float32):
    if os.path.exists(depth_path):
        depth_map = torch.load(depth_path)
    else:
        input_image = T.ToPILImage()(input_image.squeeze())
        depth_map = prepare_depth_map(
            model, input_image, dtype=dtype, device=model.device)
        torch.save(depth_map, depth_path)
        depth_image = (((depth_map + 1.0) / 2.0) * 255).to(torch.uint8)
        T.ToPILImage()(depth_image.squeeze()).convert(
            "L").save(depth_path.replace(".pt", ".png"))

    return depth_map

@torch.no_grad()
def prepare_depth_map(model, image, depth_map=None, batch_size=1, do_classifier_free_guidance=False, dtype=torch.float32, device="cuda"):
    if isinstance(image, Image.Image):
        image = [image]
    else:
        image = list(image)

    if isinstance(image[0], Image.Image):
        width, height = image[0].size
    elif isinstance(image[0], np.ndarray):
        width, height = image[0].shape[:-1]
    else:
        height, width = image[0].shape[-2:]

    if depth_map is None:
        pixel_values = model.feature_extractor(
            images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device)
        # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
        # So we use `torch.autocast` here for half precision inference.
        context_manger = torch.autocast(
            "cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
        with context_manger:
            ret = model.depth_estimator(pixel_values)
            depth_map = ret.predicted_depth
            # depth_image = ret.depth
    else:
        depth_map = depth_map.to(device=device, dtype=dtype)

    indices = depth_map != -1
    bg_indices = depth_map == -1
    min_d = depth_map[indices].min()

    if bg_indices.sum() > 0:
        depth_map[bg_indices] = min_d - 10
        # min_d = min_d - 10

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(height // model.vae_scale_factor,
              width // model.vae_scale_factor),
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
    depth_map = depth_map.to(dtype)

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    if depth_map.shape[0] < batch_size:
        repeat_by = batch_size // depth_map.shape[0]
        depth_map = depth_map.repeat(repeat_by, 1, 1, 1)

    depth_map = torch.cat(
        [depth_map] * 2) if do_classifier_free_guidance else depth_map
    return depth_map


def get_latents_dir(latents_path, model_key):
    model_key = model_key.split("/")[-1]
    return os.path.join(latents_path, model_key)


def get_controlnet_kwargs(controlnet, x, cond, t, controlnet_cond, controlnet_scale=1.0):
    down_block_res_samples, mid_block_res_sample = controlnet(
        x,
        t,
        encoder_hidden_states=cond,
        controlnet_cond=controlnet_cond,
        return_dict=False,
    )
    down_block_res_samples = [
        down_block_res_sample * controlnet_scale
        for down_block_res_sample in down_block_res_samples
    ]
    mid_block_res_sample *= controlnet_scale
    controlnet_kwargs = {"down_block_additional_residuals": down_block_res_samples,
                         "mid_block_additional_residual": mid_block_res_sample}
    return controlnet_kwargs


def get_frame_ids(frame_range, frame_ids=None):
    if frame_ids is None:
        frame_ids = list(range(*frame_range))
    frame_ids = sorted(frame_ids)

    if len(frame_ids) > 4:
        frame_ids_str = "{} {} ... {} {}".format(
            *frame_ids[:2], *frame_ids[-2:])
    else:
        frame_ids_str = " ".join(["{}"] * len(frame_ids)).format(*frame_ids)
    print("[INFO] frame indexes: ", frame_ids_str)
    return frame_ids
