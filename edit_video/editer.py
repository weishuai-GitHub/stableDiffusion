from copy import deepcopy
import os
from typing import List, Optional, Union
import numpy as np
from diffusers import (
    # AutoencoderKL,
    DDIMScheduler,
    DDIMInverseScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline
    # UNet2DConditionModel,
)
import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection,CLIPTextModelWithProjection
from util.pnp_utils import register_attention_control, register_conv_control, register_time
from util.video_utils import get_latents_dir, load_video, save_frames,load_latent, save_video
from util.config_utils import save_config

# import inspect
from edit_video.patch import apply_patch, update_patch

from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
# from masactrl.diffuser_utils import MasaCtrlPipeline
# from masactrl.masactrl_utils import AttentionBase
# from masactrl.masactrl_utils import regiter_attention_editor_diffusers
class SDEditer(StableDiffusionPipeline):
    
    def __init__(self, config,vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor, image_encoder: CLIPVisionModelWithProjection = None,requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, image_encoder,requires_safety_checker)
        self.scheduler.set_timesteps(config.generation.n_timesteps)
        # self.timesteps = self.scheduler.timesteps[20:]
        self.timesteps = self.scheduler.timesteps
        self.set_config(config)

    def set_config(self,config):
        self.edit_config = config
        self.model_key = config.model_key
        self.timesteps_to_save = self.timesteps
        # invsersion config
        inv_config = config.inversion
        self.inv_prompt=inv_config.prompt
        self.n_frames = inv_config.n_frames
        self.frame_height, self.frame_width = config.height, config.width
        self.work_dir = config.work_dir
        self.save_latents=inv_config.save_intermediate
        self.steps=inv_config.steps
        self.batch_size = inv_config.batch_size

        self.seed = config.seed
        # edit config
        gene_config = config.generation
        self.control = gene_config.control
        if self.dtype == torch.float16:
            print("[INFO] float precision fp16. Use torch.float16.")
        else:
            print("[INFO] float precision fp32. Use torch.float32.")
        self.gen_batch_size = 2
        self.n_timesteps = gene_config.n_timesteps
        self.use_pnp = self.control == "pnp"
        self.chunk_size = gene_config.chunk_size
        self.chunk_ord = gene_config.chunk_ord
        self.merge_global = gene_config.merge_global
        self.local_merge_ratio = gene_config.local_merge_ratio
        self.global_merge_ratio = gene_config.global_merge_ratio
        self.global_rand = gene_config.global_rand
        self.align_batch = gene_config.align_batch

        self.gen_prompt = gene_config.prompt
        self.negative_prompt = gene_config.negative_prompt
        self._guidance_scale = gene_config.guidance_scale
        self.save_frame = gene_config.save_frame

        self.frame_height, self.frame_width = config.height, config.width
        self.work_dir = config.work_dir

        self.chunk_ord = gene_config.chunk_ord
        if "mix" in self.chunk_ord:
            self.perm_div = float(self.chunk_ord.split("-")[-1]) if "-" in self.chunk_ord else 3.
            self.chunk_ord = "mix"
            
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=None):
        # text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
        #                             truncation=True, return_tensors='pt')
        # text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # if negative_prompt is not None:
        #     uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
        #                                   return_tensors='pt')
        #     uncond_embeddings = self.text_encoder(
        #         uncond_input.input_ids.to(self.device))[0]
        image_embeds, uncond_image_embeds = self.encode_prompt(prompt,self.device,1,
                                                               self.do_classifier_free_guidance,
                                                               negative_prompt=negative_prompt)
        text_embeddings = torch.cat([uncond_image_embeds, image_embeds])
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            imgs = 2 * imgs - 1
            imgs = imgs.to(dtype=self.vae.dtype)
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * self.vae.config.scaling_factor
        return latents

    def encode_imgs_batch(self, imgs):
        latents = []
        batch_imgs = imgs.split(self.batch_size, dim = 0)
        for img in batch_imgs:
            latents += [self.encode_imgs(img)]
        latents = torch.cat(latents)
        return latents
    
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
        return image
    
    @torch.no_grad()
    def decode_latents_batch(self, latents):
        imgs = []
        batch_latents = latents.split(self.batch_size, dim = 0)
        for latent in batch_latents:
            imgs += [self.decode_latents(latent)]
        imgs = torch.cat(imgs)
        return imgs
    
    @torch.no_grad()
    def pred_noise(self, x, cond, t ,is_guidance_scale=True,is_inversion =False,batch_idx=None):
        flen = len(x)
        if is_guidance_scale:
            text_embed_input = cond.repeat_interleave(flen, dim=0)
            # For classifier-free guidance
            latent_model_input = torch.cat([x, x])
            batch_size = 2
            # if self.use_pnp:
            #     # Cat latents from inverted source frames for PnP operation
            #     source_latents = self.cur_latents
            #     if batch_idx is not None:
            #         source_latents = source_latents[batch_idx]
            #     latent_model_input = torch.cat([source_latents.to(x), latent_model_input])
            #     batch_size += 1
            # Pred noise!
            eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample
            noise_pred_uncond, noise_pred_cond = eps.chunk(batch_size)[-2:]
            # CFG
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            _, cond = cond.chunk(2)
            latent_model_input = x
            text_embed_input = cond.repeat_interleave(flen, dim=0)
            eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample
            noise_pred = eps
        # if is_inversion:
        #     # Get the target for loss depending on the prediction type
        #     if self.scheduler.config.prediction_type == "epsilon":
        #         # noise_pred = noise_pred
        #         pass
        #     elif self.scheduler.config.prediction_type == "v_prediction":
        #         noise_pred = self.scheduler.get_velocity(x, noise_pred, t)
        #     else:
        #         raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        return noise_pred
        
    def ddim_insersion_step(self,model_output,timestep,sample,i):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        
        # 1. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[timestep] #if timestep >= 0 else self.scheduler.initial_alpha_cumprod
        # alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep]
        timesteps = reversed(self.timesteps)
        alpha_prod_t = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )

        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 3. Clip or threshold "predicted x_0"
        if self.scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range
            )

        # 4. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 5. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        return prev_sample

    def pred_next_x(self, x, eps, t, i, inversion=False):
        if inversion:
            timesteps = reversed(self.timesteps)
        else:
            timesteps = self.timesteps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        if inversion:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else self.scheduler.final_alpha_cumprod
            )
        else:
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[timesteps[i + 1]]
                if i < len(timesteps) - 1
                else self.scheduler.final_alpha_cumprod
            )
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        if inversion:
            pred_x0 = (x - sigma_prev * eps) / mu_prev
            x = mu * pred_x0 + sigma * eps
        else:
            pred_x0 = (x - sigma * eps) / mu
            x = mu_prev * pred_x0 + sigma_prev * eps

        return x
    
    @torch.no_grad()
    def ddim_inversion(self, x, conds, save_path):
        print("[INFO] start DDIM Inversion!")
        timesteps = reversed(self.timesteps)
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))
                batches = x_index.split(self.batch_size, dim = 0)
                for batch in batches:
                    noise = self.pred_noise(
                        x[batch], conds, t,False,is_inversion = True)
                    noises += [noise]
                noises = torch.cat(noises)
                # add_noise x_t -> x_t+1
                x = self.ddim_insersion_step(noises, t, x,i)
                if self.save_latents and t in self.timesteps_to_save:
                    torch.save(x, os.path.join(
                        save_path, f'noisy_latents_{t}.pt'))

        # Save inverted noise latents
        pth = os.path.join(save_path, f'noisy_latents_{t}.pt')
        torch.save(x, pth)
        print(f"[INFO] inverted latent saved to: {pth}")
        return x
    
    @torch.no_grad()
    def ddim_sample(self, x, conds):
        print("[INFO] reconstructing frames...")
        timesteps = self.timesteps
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))
                batches = x_index.split(self.batch_size, dim = 0)
                for batch in batches:
                    noise = self.pred_noise(
                        x[batch], conds, t,False)
                    noises += [noise]
                noises = torch.cat(noises)
                # compute the previous noisy sample x_t -> x_t-1
                x = self.scheduler.step(noises, t, x, return_dict=False)[0]
        return x
    
    def check_latent_exists(self, save_path):
        save_timesteps = [self.timesteps[0]]
        if self.save_latents:
            save_timesteps += self.timesteps_to_save
        for ts in save_timesteps:
            latent_path = os.path.join(
                save_path, f'noisy_latents_{ts}.pt')
            if not os.path.exists(latent_path):
                return False
        return True
    
    def prepare_cond(self, prompts, n_frames):
        if isinstance(prompts, str):
            conds = self.get_text_embeds(prompts)
            # conds = torch.cat([cond] * n_frames)
        elif isinstance(prompts, list):
            cond_ls = []
            for prompt in prompts:
                cond = self.get_text_embeds(prompt)
                cond_ls += [cond]
            conds = torch.cat(cond_ls)
        return conds, prompts

    def inversion(self,data_path, save_path):
        # self.scheduler.set_timesteps(self.steps)
        save_path = get_latents_dir(save_path, self.model_key)
        os.makedirs(save_path, exist_ok = True)
        if self.check_latent_exists(save_path) and not self.edit_config.inversion.force:
            print(f"[INFO] inverted latents exist at: {save_path}. Skip inversion! Set 'inversion.force: True' to invert again.")
            return

        frames = load_video(data_path, self.frame_height, self.frame_width, device = self.device)

        frame_ids = list(range(len(frames)))
        if self.n_frames is not None:
            frame_ids = frame_ids[:self.n_frames]
        frames = frames[frame_ids]

        conds, prompts = self.prepare_cond(self.inv_prompt, len(frames))
        with open(os.path.join(save_path, 'inversion_prompts.txt'), 'w') as f:
            f.write('\n'.join(prompts))

        latents = self.encode_imgs_batch(frames)
        torch.cuda.empty_cache()
        print(f"[INFO] clean latents shape: {latents.shape}")

        inverted_x = self.ddim_inversion(latents, conds, save_path)
        save_config(self.edit_config, save_path, inv = True)
        if self.edit_config.inversion.recon:
            latent_reconstruction = self.ddim_sample(inverted_x, conds)

            torch.cuda.empty_cache()
            recon_frames = self.decode_latents_batch(
                latent_reconstruction)

            recon_save_path = os.path.join(save_path, 'recon_frames')
            save_frames(recon_frames, recon_save_path, frame_ids = frame_ids)

    def gen_check_latent_exists(self, latent_path):
        # if self.use_pnp:
        #     timesteps = self.timesteps
        # else:
        timesteps = [self.timesteps[0]]

        for ts in timesteps:
            cur_latent_path = os.path.join(
                latent_path, f'noisy_latents_{ts}.pt')
            if not os.path.exists(cur_latent_path):
                return False
        return True

    @torch.no_grad()
    def prepare_data(self, data_path, latent_path, frame_ids):
        self.frames = load_video(data_path, self.frame_height,
                                 self.frame_width, frame_ids=frame_ids, device=self.device)
        self.init_noise = load_latent(
            latent_path, t=self.timesteps[0], frame_ids=frame_ids).to(self.dtype).to(self.device)

    def init_pnp(self):
        # qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        # conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        # register_attention_control(
        #     self, qk_injection_timesteps, num_inputs=self.gen_batch_size)
        # register_conv_control(
        #     self, conv_injection_timesteps, num_inputs=self.gen_batch_size)
        # inference the synthesized image with MasaCtrl
        STEP = 4
        LAYPER = 10

        # hijack the attention module
        editor = MutualSelfAttentionControl(STEP, LAYPER)
        regiter_attention_editor_diffusers(self, editor)
   
    def activate_vidtome(self):
        apply_patch(self, self.local_merge_ratio, self.merge_global, self.global_merge_ratio, 
            seed = self.seed, batch_size = self.gen_batch_size, align_batch = self.use_pnp or self.align_batch, global_rand = self.global_rand)
    
    def get_chunks(self, flen):
        x_index = torch.arange(flen)

        # The first chunk has a random length
        rand_first = np.random.randint(0, self.chunk_size) + 1
        chunks = x_index[rand_first:].split(self.chunk_size, dim=0)
        chunks = [x_index[:rand_first]] + list(chunks) if len(chunks[0]) > 0 else [x_index[:rand_first]]
        if np.random.rand() > 0.5:
            chunks = chunks[::-1]
        
        # Chunk order only matter when we do global token merging
        if self.merge_global == False:
            return chunks

        # Chunk order. "seq": sequential order. "rand": full permutation. "mix": partial permutation.
        if self.chunk_ord == "rand":
            order = torch.randperm(len(chunks))
        elif self.chunk_ord == "mix":
            randord = torch.randperm(len(chunks)).tolist()
            rand_len = int(len(randord) / self.perm_div)
            seqord = sorted(randord[rand_len:])
            if rand_len > 0:
                randord = randord[:rand_len]
                if abs(seqord[-1] - randord[-1]) < abs(seqord[0] - randord[-1]):
                    seqord = seqord[::-1]
                order = randord + seqord
            else:
                order = seqord
        else:
            order = torch.arange(len(chunks))
        chunks = [chunks[i] for i in order]
        return chunks
    
    def pre_iter(self, x, t):
        pass
        # if self.use_pnp:
        #     # Prepare PnP
        #     register_time(self, t.item())
        #     cur_latents = load_latent(self.latent_path, t=t, frame_ids = self.frame_ids)
        #     self.cur_latents = cur_latents

    def post_iter(self, x, t):
        if self.merge_global:
            # Reset global tokens
            update_patch(self, global_tokens = None)
    
    @torch.no_grad()
    def gen_pred_noise(self, x, cond, t ,is_guidance_scale=True,is_inversion =False,batch_idx=None):
        # flen = len(x)
        # if is_guidance_scale:
        text_embed_input = cond
        # For classifier-free guidance
        latent_model_input = x
        # Pred noise!
        eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample
        
        # CFG
        if is_guidance_scale:
            noise_pred_uncond, noise_pred_cond = eps.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = eps
        # else:
        #     _, cond = cond.chunk(2)
        #     if self.use_pnp:
        #         latent_model_input = torch.cat([x, x])
        #     else:latent_model_input = x
        #     text_embed_input = cond.repeat_interleave(flen, dim=0)
        #     eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input).sample
        #     noise_pred = eps
        # if is_inversion:
        #     # Get the target for loss depending on the prediction type
        #     if self.scheduler.config.prediction_type == "epsilon":
        #         # noise_pred = noise_pred
        #         pass
        #     elif self.scheduler.config.prediction_type == "v_prediction":
        #         noise_pred = self.scheduler.get_velocity(x, noise_pred, t)
        #     else:
        #         raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        return noise_pred
    @torch.no_grad()
    def gen_ddim_sample(self, x, conds):
        print("[INFO] denoising frames...")
        timesteps = self.timesteps
        flen = len(x)
        if self.use_pnp:
            x = torch.cat([x,x])
        
        conds = conds.repeat_interleave(flen, dim=0)
        noises = torch.zeros_like(x)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            self.pre_iter(x, t)

            # Split video into chunks and denoise
            chunks = self.get_chunks(len(x))
            for chunk in chunks:
                torch.cuda.empty_cache()
                if self.do_classifier_free_guidance:
                    text_embed_input = torch.cat([cond[chunk] for cond in conds.chunk(2)])
                    latent_model_input = torch.cat([x[chunk] for _ in range(2)])
                else:
                    text_embed_input = conds[chunk]
                    latent_model_input = x[chunk]
                noises[chunk] = self.gen_pred_noise(
                    latent_model_input, text_embed_input, t, is_guidance_scale=self.do_classifier_free_guidance)
            x = self.scheduler.step(noises, t, x, return_dict=False)[0]

            self.post_iter(x, t)
        return x

    def generate(self,data_path, latent_path, output_path, frame_ids):
        self.activate_vidtome()
        if self.use_pnp:
            self.init_pnp()
        self.scheduler.set_timesteps(self.n_timesteps)
        latent_path = get_latents_dir(latent_path, self.model_key)

        assert self.gen_check_latent_exists(
            latent_path), f"Required latent not found at {latent_path}. \
                    Note: If using PnP as control, you need inversion latents saved \
                     at each generation timestep."
        
        self.data_path = data_path
        self.latent_path = latent_path
        self.frame_ids = frame_ids
        self.prepare_data(data_path, latent_path, frame_ids)

        print(f"[INFO] initial noise latent shape: {self.init_noise.shape}")

        for edit_name, edit_prompt in self.gen_prompt.items():
            print(f"[INFO] current prompt: {edit_prompt}")
            if self.use_pnp:
                conds = self.get_text_embeds([self.inv_prompt,edit_prompt], [self.negative_prompt,self.negative_prompt])
                clean_latent = self.gen_ddim_sample(self.init_noise, conds)
            else:
                conds = self.get_text_embeds(edit_prompt, self.negative_prompt)
                # Comment this if you have enough GPU memory
                clean_latent = self.gen_ddim_sample(self.init_noise, conds)
            torch.cuda.empty_cache()
            clean_frames = self.decode_latents_batch(clean_latent)
            cur_output_path = os.path.join(output_path, edit_name)
            save_config(self.edit_config, cur_output_path, gene = True)
            save_video(clean_frames, cur_output_path, save_frame = self.save_frame)

if __name__ =='__main__':
    from util.config_utils import load_config
    from util.video_utils import seed_everything, get_frame_ids
    repo_id = "runwayml/stable-diffusion-v1-5"
    config = load_config()
    config.model_key = repo_id
    seed_everything(config.seed)
    device = 'cuda:0'
    pipe = StableDiffusionPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
    pipe.to(device)
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    unet = pipe.unet
    scheduler = pipe.scheduler
    image_encoder = pipe.image_encoder
    feature_extractor = pipe.feature_extractor
    
    edit = SDEditer(config,vae, text_encoder,tokenizer, unet, 
                    scheduler, None, feature_extractor,image_encoder,False)
    # edit.to(device)
    print("Start inversion!")

    edit.inversion(config.input_path, config.inversion.save_path)

    print("Start generation!")
    frame_ids = get_frame_ids(
        config.generation.frame_range, config.generation.frame_ids)
    edit.generate(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)