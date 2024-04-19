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
# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection,CLIPTextModelWithProjection
from util.pnp_utils import register_attention_control, register_conv_control, register_time
from util.video_utils import get_latents_dir, load_video, save_frames,load_latent, save_video
from util.config_utils import save_config

import inspect
from edit_video.patch import apply_patch, update_patch

# from masactrl.diffuser_utils import MasaCtrlPipeline
# from masactrl.masactrl_utils import AttentionBase
# from masactrl.masactrl_utils import regiter_attention_editor_diffusers
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class SDXLEditer(StableDiffusionXLPipeline):
    
    def __init__(self,
        config,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,):
        super().__init__(vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler, image_encoder, feature_extractor, force_zeros_for_empty_prompt, add_watermarker)
        self.scheduler.set_timesteps(config.generation.n_timesteps)
        self.set_config(config)

    def set_config(self,config):
        self.edit_config = config
        self.model_key = config.model_key
        self.timesteps_to_save = self.scheduler.timesteps


        # invsersion config
        inv_config = config.inversion
        self.inv_config=inv_config
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
        if self.use_pnp:
            pnp_f_t = int(gene_config.n_timesteps * gene_config.pnp_f_t)
            pnp_attn_t = int(gene_config.n_timesteps * gene_config.pnp_attn_t)
            self.gen_batch_size += 1
            self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
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
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(prompt,prompt_2=None,
                            device=self.device,
                            negative_prompt=negative_prompt,
                            )
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        
        add_time_ids = self._get_add_time_ids(
            (self.frame_height, self.frame_width),
            (0,0),
            (self.frame_height, self.frame_height),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds],dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds],dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        add_time_ids  = add_time_ids.to(self.device)
        return text_embeddings,add_text_embeds,add_time_ids

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # dtype=self.dtype
        if self.vae.config.force_upcast:
            imgs = imgs.float()
            self.vae.to(dtype=torch.float32)
        imgs = 2 * imgs - 1

        imgs = imgs.to(dtype=self.vae.dtype)
        posterior = self.vae.encode(imgs).latent_dist.sample()
        latents = posterior* self.vae.config.scaling_factor
        return latents.to(self.dtype)

    def encode_imgs_batch(self, imgs):
        latents = []
        batch_imgs = imgs.split(self.batch_size, dim = 0)
        for img in batch_imgs:
            latents += [self.encode_imgs(img)]
        latents = torch.cat(latents)
        return latents
    
    def decode_latents(self, latents):
        if self.vae.config.force_upcast:
            latents = latents.float()
            self.vae.to(dtype=torch.float32)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents,return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.to(self.dtype)
    
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
        text_embeddings,add_text_embeds,add_time_ids = cond
        if is_guidance_scale:
            text_embeddings = text_embeddings.repeat_interleave(flen, dim=0)
            add_text_embeds = add_text_embeds.repeat_interleave(flen, dim=0)
            add_time_ids = add_time_ids.repeat_interleave(flen, dim=0)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            # For classifier-free guidance
            latent_model_input = torch.cat([x, x])
            batch_size = 2
            if self.use_pnp:
                # Cat latents from inverted source frames for PnP operation
                source_latents = self.cur_latents
                if batch_idx is not None:
                    source_latents = source_latents[batch_idx]
                latent_model_input = torch.cat([source_latents.to(x), latent_model_input])
                batch_size += 1
            # Pred noise!
            eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,
                            added_cond_kwargs=added_cond_kwargs,).sample
            noise_pred_uncond, noise_pred_cond = eps.chunk(2)
            # CFG
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            _, text_embeddings = text_embeddings.chunk(2)
            _,add_text_embeds = add_text_embeds.chunk(2)
            _,add_time_ids = add_time_ids.chunk(2)
            latent_model_input = x
            text_embed_input = text_embeddings.repeat_interleave(flen, dim=0)
            add_text_embeds = add_text_embeds.repeat_interleave(flen, dim=0)
            add_time_ids = add_time_ids.repeat_interleave(flen, dim=0)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            eps = self.unet(latent_model_input, t, 
                            encoder_hidden_states=text_embed_input,
                            added_cond_kwargs = added_cond_kwargs).sample
            noise_pred = eps
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
        timesteps = reversed(self.scheduler.timesteps)
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
            timesteps = reversed(self.scheduler.timesteps)
        else:
            timesteps = self.scheduler.timesteps
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
    
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def ddim_inversion(self, x, conds, save_path):
        print("[INFO] start DDIM Inversion!")
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, config.generation.n_timesteps, device, None)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            0.3,
            device,)
        latent_timestep = timesteps[:1].repeat(x.shape[0])
        self.timesteps = timesteps
        noises = randn_tensor(x.shape, device=self.device, dtype=self.dtype)
        # z_t -> z_(t-1)
        x = self.scheduler.add_noise(x, noises, latent_timestep)
        # Save inverted noise latents
        pth = os.path.join(save_path, f'noisy_latents_{latent_timestep[0].item()}.pt')
        torch.save(x, pth)
        print(f"[INFO] inverted latent saved to: {pth}")
        return x
    
    @torch.no_grad()
    def ddim_sample(self, x, conds):
        print("[INFO] reconstructing frames...")
        # g = torch.Generator('cuda').manual_seed(1)
        # timesteps = self.scheduler.timesteps
        # x = randn_tensor(x.shape, device=self.device, dtype=self.dtype)
        # x = x* self.scheduler.init_noise_sigma
        timesteps = self.timesteps
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            for i, t in enumerate(tqdm(timesteps)):
                noises = []
                x_index = torch.arange(len(x))
                batches = x_index.split(self.batch_size, dim = 0)
                for batch in batches:
                    latent_model_input = self.scheduler.scale_model_input(x[batch], t)
                    noise = self.pred_noise(
                        latent_model_input, conds, t,False)
                    noises += [noise]
                noises = torch.cat(noises)
                # compute the previous noisy sample x_t -> x_t-1
                x = self.scheduler.step(noises, t, x, return_dict=False)[0]
        return x
    
    def check_latent_exists(self, save_path):
        save_timesteps = [self.scheduler.timesteps[0]]
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
            text_embeddings,add_text_embeds,add_time_ids = self.get_text_embeds(prompts)
            # conds = torch.cat([cond] * n_frames)
        conds = (text_embeddings,add_text_embeds,add_time_ids)
        return conds, prompts

    def inversion(self,data_path, save_path):
        self.scheduler.set_timesteps(self.steps)
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
        if self.use_pnp:
            timesteps = self.timesteps
        else:
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

    def init_pnp(self, conv_injection_t, qk_injection_t):
        qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control(
            self, qk_injection_timesteps, num_inputs=self.gen_batch_size)
        register_conv_control(
            self, conv_injection_timesteps, num_inputs=self.gen_batch_size)
        
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
        if self.use_pnp:
            # Prepare PnP
            register_time(self, t.item())
            cur_latents = load_latent(self.latent_path, t=t, frame_ids = self.frame_ids)
            self.cur_latents = cur_latents

    def post_iter(self, x, t):
        if self.merge_global:
            # Reset global tokens
            update_patch(self, global_tokens = None)

    @torch.no_grad()
    def gen_ddim_sample(self, x, conds):
        print("[INFO] denoising frames...")
        noises = torch.zeros_like(x)
        
        for i, t in enumerate(tqdm(self.timesteps, desc="Sampling")):
            self.pre_iter(x, t)
            # Split video into chunks and denoise
            chunks = self.get_chunks(len(x))
            for chunk in chunks:
                torch.cuda.empty_cache()
                latent_model_input = self.scheduler.scale_model_input(x[chunk], t)
                noises[chunk] = self.pred_noise(
                    latent_model_input, conds, t, is_guidance_scale=self.do_classifier_free_guidance)

            x = self.scheduler.step(noises, t, x, return_dict=False)[0]

            self.post_iter(x, t)
        return x

    def generate(self,data_path, latent_path, output_path, frame_ids):
        # self.activate_vidtome()
        latent_path = get_latents_dir(latent_path, self.model_key)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, config.generation.n_timesteps, device, None)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            0.3,
            device,)
        self.timesteps = timesteps
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
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    config = load_config()
    config.model_key = repo_id
    seed_everything(config.seed)
    device = 'cuda:0'
    pipe = StableDiffusionXLPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
    pipe.to(device)
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    unet = pipe.unet
    scheduler = pipe.scheduler
    image_encoder = pipe.image_encoder
    feature_extractor = pipe.feature_extractor
    
    edit = SDXLEditer(config,vae, text_encoder,text_encoder_2,tokenizer,tokenizer_2, unet, scheduler, image_encoder, feature_extractor)
    # edit.to(device)
    print("Start inversion!")

    edit.inversion(config.input_path, config.inversion.save_path)

    print("Start generation!")
    frame_ids = get_frame_ids(
        config.generation.frame_range, config.generation.frame_ids)
    edit.generate(config.input_path, config.generation.latents_path,
              config.generation.output_path, frame_ids=frame_ids)