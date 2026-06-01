# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonolayers.github.io
# --------------------------------------------------------------------------


import logging
from typing import Optional, Union

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from src.util.lrm import LRM
from VAE.vae import RR_Decoder2, RR_Encoder

from .util.batchsize import find_batch_size
from .util.image_util import chw2hwc, get_tv_resample_method, resize_max_res


def cat_hidden_states(hidden_states):
    """Concatenate each layer's attention stream with the other layer's copy
    along the feature dimension, enabling cross-layer attention."""
    B, C, L = hidden_states.shape
    hidden_states_cat = hidden_states.new_zeros(B, C * 2, L)
    hidden_states_cat[:, :C, :] = hidden_states
    hidden_states_cat[:B // 2, C:, :] = hidden_states[B // 2:, :, :]
    hidden_states_cat[B // 2:, C:, :] = hidden_states[:B // 2, :, :]
    return hidden_states_cat


def get_perpendicular_component(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the component of `x` perpendicular to `y`.

    Computes ``x - proj_y(x)``, where ``proj_y(x) = ((x·y) / ||y||^2) * y``.
    """
    assert x.shape == y.shape
    return x - ((x * y).sum() / (torch.norm(y) ** 2)) * y


def apply_separation_sampling(
    t_pred: torch.Tensor,
    r_pred: torch.Tensor,
    scale: float = 0.2,
):
    """Push each layer's noise prediction away from the other's direction.

    For each layer, subtract the projection of the other layer's noise onto
    this one, then add a ``scale``-weighted copy of the residual — a
    perpendicular-direction pull that discourages the two streams from
    predicting overlapping noise.
    """
    t_prev, r_prev = t_pred.clone(), r_pred.clone()
    t_pred = t_pred + scale * (t_pred - get_perpendicular_component(r_prev, t_pred))
    r_pred = r_pred + scale * (r_pred - get_perpendicular_component(t_prev, r_pred))
    return t_pred, r_pred


class LayerAttnProcessor:
    """Attention processor that cross-wires transmission and reflection streams.

    When it runs as self-attention (``encoder_hidden_states is None``), keys and
    values are concatenated with the corresponding tensors from the other layer
    along the feature dimension (see :func:`cat_hidden_states`), so each layer
    attends over both its own features and the other layer's. Cross-attention
    with text embeddings passes through unchanged.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "LayerAttnProcessor requires PyTorch 2.0; please upgrade."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        is_self_attention = encoder_hidden_states is None
        if is_self_attention:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if is_self_attention:
            # Cross-wire the other layer's k/v so each layer attends over both.
            key = cat_hidden_states(key)
            value = cat_hidden_states(value)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

def exponential_decay_list(init_weight, decay_rate, num_steps):
    """Return ``[w, w*r, w*r^2, ..., w*r^(n-1)]`` on CUDA."""
    weights = [init_weight * (decay_rate ** i) for i in range(num_steps)]
    return torch.tensor(weights).cuda()


def prepare_unet(unet):
    """Swap every attention processor for :class:`LayerAttnProcessor` and freeze weights."""
    attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        attn_procs[name] = LayerAttnProcessor()
        module.requires_grad_(False)
    unet.set_attn_processor(attn_procs)
    return unet


class LayerSepOutput(BaseOutput):
    """Output of :class:`LayersepPipeline`.

    Fields:
        transmission_np / reflection_np: decoded layers as float arrays in [0, 1].
        transmission_Image / reflection_Image: same content as PIL images.
        ori_transmission_Image: transmission before the optional CFW refiner pass.
        pseudo_M: decoded reconstruction of the input (``LRM(T, R)`` latents) as a PIL image.
    """
    transmission_np: np.ndarray
    reflection_np: np.ndarray
    transmission_Image: Union[None, Image.Image]
    reflection_Image: Union[None, Image.Image]
    ori_transmission_Image: Union[None, Image.Image]
    pseudo_M: Union[None, Image.Image]


class LayersepPipeline(DiffusionPipeline):
    """Diffusion pipeline for layer separation / reflection removal.

    Inherits from :class:`~diffusers.DiffusionPipeline`.

    Given an RGB image that is a superposition of a transmission layer and a
    reflection layer, the pipeline runs a two-stream DDIM denoising loop (the
    UNet's conv-in has been widened to accept concatenated per-layer latents,
    and its self-attention is replaced by :class:`LayerAttnProcessor` so the
    two streams can attend to each other). Optional latent optimization and
    separation sampling sharpen the split. Decoding uses a CFW-refined VAE
    decoder.

    Args:
        unet: widened UNet conditioned on class labels (0=transmission, 1=reflection).
        vae: autoencoder whose decoder is replaced by :class:`RR_Decoder2` in :meth:`set_decoder`.
        scheduler: DDIM or LCM scheduler.
        text_encoder / tokenizer: CLIP pair used to embed the prompts
            ``"transmission"`` and ``"reflection"``.
        default_denoising_steps: falls back to the pipeline default when the
            caller does not pass ``denoising_steps``.
        default_processing_resolution: same, for ``processing_res``.
    """

    rgb_latent_scale_factor = 0.18215
    layer_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

    def set_decoder(self):
        """Upgrade the VAE to the reflection-removal encoder / CFW decoder."""
        self.vae.encoder.__class__ = RR_Encoder
        self.vae.decoder = RR_Decoder2(self.vae.decoder)

    def _load_lrm_checkpoints(self, lrm_checkpoints_path, device):
        """Load the latent reconstruction module used by ``--optimization``."""
        self.lrm = LRM(4)
        state_dict = torch.load(lrm_checkpoints_path)
        self.lrm.load_state_dict(state_dict)
        self.lrm = self.lrm.to(device)

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Optional[torch.Generator] = None,
        show_progress_bar: bool = True,
        cross_attention: bool = True,
        inf_res: int = 512,
        w: float = 1.0,
        optimization: bool = False,
        s_sampling: bool = False,
    ) -> LayerSepOutput:
        """Split `input_image` into a transmission layer and a reflection layer.

        Args:
            input_image: RGB image, as PIL or ``[1, 3, H, W]`` tensor.
            denoising_steps: DDIM denoising steps. At least 10 is recommended;
                1-4 for LCM checkpoints. Falls back to
                ``self.default_denoising_steps``.
            ensemble_size: Number of predictions to ensemble.
            processing_res: Maximum edge length for preprocessing resize; 0
                keeps native resolution. Falls back to
                ``self.default_processing_resolution``.
            match_input_res: Resize outputs back to input resolution.
            resample_method: ``bilinear`` | ``bicubic`` | ``nearest``.
            batch_size: Inference batch size (≤ ``ensemble_size``). ``0`` lets
                the pipeline pick.
            generator: Generator for initial noise.
            show_progress_bar: Display a tqdm bar over denoising steps.
            cross_attention: Install :class:`LayerAttnProcessor` on the UNet.
            inf_res: Operating square resolution (both sides resized to this).
            w: CFW refiner blend weight used when decoding the transmission.
            optimization: Enable the inner latent-optimization loop.
            s_sampling: Enable separation sampling on the noise predictions.

        Returns:
            :class:`LayerSepOutput` containing the two layers plus the
            pre-refiner transmission and a reconstructed ``pseudo_M`` image.
        """
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1
        if cross_attention:
            self._modify_ca()
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image preprocess -----------------
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image).unsqueeze(0)  # [1, 3, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image)!r}")
        assert rgb.dim() == 4 and rgb.shape[-3] == 3, (
            f"Wrong input shape {tuple(rgb.shape)}, expected [1, 3, H, W]"
        )

        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        rgb = (rgb / 255.0).to(self.device)
        _, _, H, W = rgb.shape
        rgb = resize(rgb, (inf_res, inf_res), resample_method, antialias=True)
        rgb_norm = (rgb * 2.0 - 1.0).to(self.dtype)  # [0, 1] -> [-1, 1]

        # ----------------- Ensemble-wise denoising -----------------
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        effective_bs = batch_size if batch_size > 0 else find_batch_size(
            ensemble_size=ensemble_size,
            input_res=max(rgb_norm.shape[1:]),
            dtype=self.dtype,
        )
        loader = DataLoader(
            TensorDataset(duplicated_rgb), batch_size=effective_bs, shuffle=False
        )

        transmission_pred_ls, reflection_pred_ls = [], []
        ori_transmission_pred_ls, pseudo_M_pred_ls = [], []

        iterable = (
            tqdm(loader, desc="  Inference batches", leave=False)
            if show_progress_bar
            else loader
        )
        for (batched_img,) in iterable:
            transmission_latent, reflection_latent, int_M_results, pseudo_M_latent = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                batch_size=batch_size,
                optimization=optimization,
                s_sampling=s_sampling,
            )
            ori_transmission = self.shift_to_01(self.decode_layer(transmission_latent))
            transmission = self.shift_to_01(self.decode_layer(transmission_latent, int_M_results, w))
            reflection = self.shift_to_01(self.decode_layer(reflection_latent))
            pseudo_M = self.shift_to_01(self.decode_layer(pseudo_M_latent))

            transmission_pred_ls.append(transmission.detach())
            reflection_pred_ls.append(reflection.detach())
            ori_transmission_pred_ls.append(ori_transmission.detach())
            pseudo_M_pred_ls.append(pseudo_M.detach())

        ori_transmission_preds = torch.cat(ori_transmission_pred_ls, dim=0)
        transmission_preds = torch.cat(transmission_pred_ls, dim=0)
        reflection_preds = torch.cat(reflection_pred_ls, dim=0)
        pseudo_M_preds = torch.cat(pseudo_M_pred_ls, dim=0)
        torch.cuda.empty_cache()

        # Restore the square processing-resolution outputs to the input
        # resolution (also undoes the aspect-ratio squish from the square
        # resize at line 360). Area resampling for the common downsampling
        # case; bilinear (antialiased) when the input is larger than inf_res.
        if match_input_res:
            def _resize_back(t: torch.Tensor) -> torch.Tensor:
                if tuple(t.shape[-2:]) == (H, W):
                    return t
                if H <= t.shape[-2] and W <= t.shape[-1]:
                    return F.interpolate(t, size=(H, W), mode="area")
                return F.interpolate(
                    t, size=(H, W), mode="bilinear", align_corners=False, antialias=True
                )

            ori_transmission_preds = _resize_back(ori_transmission_preds)
            transmission_preds = _resize_back(transmission_preds)
            reflection_preds = _resize_back(reflection_preds)
            pseudo_M_preds = _resize_back(pseudo_M_preds)

        def _finalize(t: torch.Tensor) -> np.ndarray:
            return t.squeeze().cpu().numpy().clip(0, 1)

        ori_transmission_preds = _finalize(ori_transmission_preds)
        transmission_preds = _finalize(transmission_preds)
        reflection_preds = _finalize(reflection_preds)
        pseudo_M_preds = _finalize(pseudo_M_preds)

        return LayerSepOutput(
            transmission_np=transmission_preds,
            reflection_np=reflection_preds,
            transmission_Image=self._to_img(transmission_preds),
            reflection_Image=self._to_img(reflection_preds),
            ori_transmission_Image=self._to_img(ori_transmission_preds),
            pseudo_M=self._to_img(pseudo_M_preds),
        )

    def _load_refiner_checkpoints(self, refiner_checkpoints_path, device):
        """Load CFW fuse-block weights into the VAE decoder."""
        state_dict = torch.load(refiner_checkpoints_path)
        self.vae.decoder.fuse_blocks.load_state_dict(state_dict)
        self.vae = self.vae.to(device)

    def _modify_ca(self):
        """Install :class:`LayerAttnProcessor` on every UNet attention module."""
        self.unet = prepare_unet(self.unet)

    def _to_img(self, layer: np.ndarray) -> Image.Image:
        """Convert a ``[C, H, W]`` float array in [0, 1] to a PIL ``Image``."""
        layer = (layer.squeeze() * 255).astype(np.uint8)
        return Image.fromarray(chw2hwc(layer))

    def _set_class_embedding(self):
        """Add a 2-way class embedding to the UNet (0=transmission, 1=reflection)."""
        self.unet._set_class_embedding(
            class_embed_type=None,
            act_fn=None,
            num_class_embeds=2,
            projection_class_embeddings_input_dim=None,
            time_embed_dim=1280,
            timestep_input_dim=None,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """Warn if the requested denoising step count is off-spec for the scheduler."""
        assert n_step >= 1
        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal denoising steps: {n_step}. LCM recommends 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def _encode_prompt(
        self,
        prompt: str,
        device: torch.device,
        bs: int,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        num_images_per_prompt: int = 1,
    ) -> torch.Tensor:
        """Encode a text prompt to CLIP embeddings, tiled to batch size ``bs``."""
        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids
            if (untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
                    and not torch.equal(text_input_ids, untruncated_ids)):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1:-1]
                )
                logging.warning(
                    "Input truncated; CLIP max seq length %d. Removed: %s",
                    self.tokenizer.model_max_length, removed_text,
                )

            use_mask = getattr(self.text_encoder.config, "use_attention_mask", False)
            attention_mask = text_inputs.attention_mask.to(device) if use_mask else None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask,
            )[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # Tile: first along num_images_per_prompt, then along the outer batch bs.
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_embeds = prompt_embeds.repeat(bs, 1, 1)
        return prompt_embeds

    # ----- constants for the inner latent-optimization loop -----
    OPT_EVERY_N_STEPS = 5    # run the optimization block every N denoising steps
    OPT_INNER_ITERS = 4      # number of gradient iterations per optimization call
    OPT_LR = 0.001
    OPT_LOSS_SCALE = 10.0    # multiplier applied to the composition MSE
    SYNC_INIT_WEIGHT = 0.1   # step size for the gradient update on latents
    SYNC_DECAY_RATE = 0.99

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Optional[torch.Generator],
        show_pbar: bool,
        batch_size: int,
        optimization: bool = False,
        s_sampling: bool = False,
    ):
        """Run one DDIM pass of layer separation on a single batched image.

        Returns ``(transmission_latent, reflection_latent, int_M_results,
        pseudo_M_latent)``.

        When ``optimization`` is True, every ``OPT_EVERY_N_STEPS`` steps the
        method runs ``OPT_INNER_ITERS`` inner iterations that nudge both
        latents so that ``LRM(T, R)`` matches the encoded RGB latent
        (composition constraint). When ``s_sampling`` is True, each step's
        noise prediction is pushed away from the other layer's perpendicular
        component (separation sampling).
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        rgb_latent, int_M_results = self.encode_rgb(rgb_in)

        # [B, 4, h, w] i.i.d. noise per layer
        noise_shape = rgb_latent.shape
        transmission_latent = torch.randn(noise_shape, device=device, dtype=self.dtype, generator=generator)
        reflection_latent = torch.randn(noise_shape, device=device, dtype=self.dtype, generator=generator)

        prompt_embed_transmission = self._encode_prompt("transmission", device, bs=batch_size)
        prompt_embed_reflection = self._encode_prompt("reflection", device, bs=batch_size)
        prompt_embeds_combined = torch.cat([prompt_embed_transmission, prompt_embed_reflection], dim=0)

        # class label: 0 = transmission, 1 = reflection
        class_labels_combined = torch.cat([
            torch.zeros(batch_size, dtype=torch.int32, device=device),
            torch.ones(batch_size, dtype=torch.int32, device=device),
        ])

        sync_scheduler = exponential_decay_list(
            init_weight=self.SYNC_INIT_WEIGHT,
            decay_rate=self.SYNC_DECAY_RATE,
            num_steps=num_inference_steps,
        )

        iterable = enumerate(timesteps)
        if show_pbar:
            iterable = tqdm(iterable, total=len(timesteps),
                            leave=False, desc="    Diffusion denoising")

        with torch.enable_grad():
            for i, t in iterable:
                transmission_unet_input = torch.cat([rgb_latent, transmission_latent], dim=1)
                reflection_unet_input = torch.cat([rgb_latent, reflection_latent], dim=1)
                cat_latents = torch.cat([transmission_unet_input, reflection_unet_input], dim=0)
                t_ori = t.clone()
                t_pair = t.repeat(2)

                if optimization and i % self.OPT_EVERY_N_STEPS == 0:
                    transmission_latent, reflection_latent = self._optimize_latents(
                        i=i,
                        t_pair=t_pair,
                        t_ori=t_ori,
                        cat_latents=cat_latents,
                        prompt_embeds=prompt_embeds_combined,
                        class_labels=class_labels_combined,
                        rgb_latent=rgb_latent,
                        transmission_latent=transmission_latent,
                        reflection_latent=reflection_latent,
                        transmission_unet_input=transmission_unet_input,
                        reflection_unet_input=reflection_unet_input,
                        sync_scheduler=sync_scheduler,
                        generator=generator,
                    )

                with torch.no_grad():
                    model_pred = self.unet(
                        sample=cat_latents,
                        timestep=t_pair,
                        encoder_hidden_states=prompt_embeds_combined,
                        class_labels=class_labels_combined,
                    ).sample
                    transmission_noise_pred, reflection_noise_pred = model_pred.chunk(2)

                    if s_sampling:
                        transmission_noise_pred, reflection_noise_pred = apply_separation_sampling(
                            transmission_noise_pred, reflection_noise_pred,
                        )

                    transmission_latent = self.scheduler.step(
                        transmission_noise_pred, t_ori, transmission_latent, generator=generator
                    ).prev_sample
                    reflection_latent = self.scheduler.step(
                        reflection_noise_pred, t_ori, reflection_latent, generator=generator
                    ).prev_sample

        if optimization:
            pseudo_M_latent = self.lrm(transmission_latent, reflection_latent)
        else:
            pseudo_M_latent = transmission_latent

        return transmission_latent, reflection_latent, int_M_results, pseudo_M_latent

    def _optimize_latents(
        self,
        *,
        i: int,
        t_pair: torch.Tensor,
        t_ori: torch.Tensor,
        cat_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        class_labels: torch.Tensor,
        rgb_latent: torch.Tensor,
        transmission_latent: torch.Tensor,
        reflection_latent: torch.Tensor,
        transmission_unet_input: torch.Tensor,
        reflection_unet_input: torch.Tensor,
        sync_scheduler: torch.Tensor,
        generator: Optional[torch.Generator],
    ):
        """Nudge ``(T, R)`` latents so that ``LRM(T, R)`` reconstructs ``rgb_latent``.

        Runs :attr:`OPT_INNER_ITERS` gradient iterations. Returns the updated
        ``(transmission_latent, reflection_latent)``.
        """
        # UNet/VAE weights must remain frozen during latent-space optimization.
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        for k in range(self.OPT_INNER_ITERS):
            logging.debug("optimization step %d.%d", i, k)
            ori_transmission_latent = transmission_latent.clone().detach()
            ori_reflection_latent = reflection_latent.clone().detach()

            for tensor in (rgb_latent, transmission_latent, reflection_latent,
                           transmission_unet_input, reflection_unet_input):
                tensor.requires_grad_(True)
                tensor.retain_grad()
            torch.set_grad_enabled(True)

            optimizer = bnb.optim.AdamW8bit(
                [transmission_unet_input, reflection_unet_input],
                lr=self.OPT_LR, weight_decay=0.0,
            )

            model_pred = self.unet(
                sample=cat_latents,
                timestep=t_pair,
                encoder_hidden_states=prompt_embeds,
                class_labels=class_labels,
            ).sample
            transmission_noise_pred, reflection_noise_pred = model_pred.chunk(2)

            transmission_latent = self.scheduler.step(
                transmission_noise_pred, t_ori, transmission_latent, generator=generator
            ).pred_original_sample
            reflection_latent = self.scheduler.step(
                reflection_noise_pred, t_ori, reflection_latent, generator=generator
            ).pred_original_sample

            pseudo_M_latent = self.lrm(transmission_latent, reflection_latent)
            composition_loss = F.mse_loss(rgb_latent, pseudo_M_latent)
            loss = self.OPT_LOSS_SCALE * composition_loss
            logging.debug("composition_loss=%.6f loss=%.6f", composition_loss.item(), loss.item())

            transmission_unet_input.retain_grad()
            transmission_latent.retain_grad()
            reflection_unet_input.retain_grad()
            reflection_latent.retain_grad()

            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)

            step_size = sync_scheduler[i]
            transmission_latent = (
                ori_transmission_latent
                - step_size * torch.norm(ori_transmission_latent) * transmission_latent.grad
            )
            reflection_latent = (
                ori_reflection_latent
                - step_size * torch.norm(ori_reflection_latent) * reflection_latent.grad
            )

        return transmission_latent, reflection_latent

    def shift_to_01(self, layer: torch.Tensor) -> torch.Tensor:
        """Clip to [-1, 1] then rescale to [0, 1]."""
        return (torch.clip(layer, -1.0, 1.0) + 1.0) / 2.0

    def encode_rgb(self, rgb_in: torch.Tensor):
        """Encode an RGB image into a latent plus the encoder's intermediate features.

        Returns ``(rgb_latent, int_M_results)``, where ``int_M_results`` are
        the hidden maps consumed by the CFW-refined decoder.
        """
        h, int_M_results = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, _logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent, int_M_results

    def decode_layer(
        self,
        layer_latent: torch.Tensor,
        int_M_results=None,
        w: float = 1.0,
    ) -> torch.Tensor:
        """Decode a layer latent to an image.

        ``int_M_results`` and ``w`` are forwarded to the CFW-refined decoder;
        passing ``None`` skips the refiner fusion.
        """
        layer_latent = layer_latent / self.layer_latent_scale_factor
        z = self.vae.post_quant_conv(layer_latent)
        return self.vae.decoder(z, int_M_results, w=w)