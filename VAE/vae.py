from diffusers.models.autoencoders.vae import Decoder, Encoder
from diffusers.utils import is_torch_version
from typing import Optional, List
import torch
import torch.nn as nn

from .CFW import Fuse_sft_block_RRDB

class RR_Encoder(Encoder):
    # def forward(self, sample: torch.Tensor) -> torch.Tensor:
    def forward(self, sample: torch.Tensor, n_residual: int = 3) -> dict:
        # print("=== Encoder Architecture ===")
        # print("Conv In:", self.conv_in)
        # print("\nDown Blocks:")
        # for i, block in enumerate(self.down_blocks):
        #     print(f"Block {i}:", block)
        # print("\nMid Block:", self.mid_block)
        # print("\nOutput Layers:")
        # print("Conv Norm Out:", self.conv_norm_out)
        # print("Conv Act:", self.conv_act) 
        # print("Conv Out:", self.conv_out)
        # print("========================")
        int_results = []
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for i, down_block in enumerate(self.down_blocks):
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                    if i < n_residual:
                        int_results.append(sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for i, down_block in enumerate(self.down_blocks):
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                    if i < n_residual:
                        int_results.append(sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for i, down_block in enumerate(self.down_blocks):
                sample = down_block(sample)
                if i < n_residual:
                    int_results.append(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        int_results.reverse()

        return sample, int_results
    

class RR_Decoder2(nn.Module):
    def __init__(self, ori_Decoder):
        super(RR_Decoder2, self).__init__()

        for name, module in ori_Decoder.named_children():
            setattr(self, name, module)
            # freeze original decoder
            for param in module.parameters():
                param.requires_grad = False

        latent_M_dim_list = [512, 256, 128]
        latent_T_dim_list = [512, 512, 512]
        # self.fuse_blocks = nn.ModuleList([fuse_block(latent_M_dim, latent_T_dim, latent_M_dim // 4) for latent_M_dim, latent_T_dim in zip(latent_M_dim_list, latent_T_dim_list)])
        self.fuse_blocks = nn.ModuleList([Fuse_sft_block_RRDB(latent_M_dim, latent_T_dim) for latent_M_dim, latent_T_dim in zip(latent_M_dim_list, latent_T_dim_list)])
    def forward(
        self,
        sample: torch.Tensor,
        int_M_results: List = None,
        latent_embeds: Optional[torch.Tensor] = None,
        w=1,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""
        if self.fuse_blocks is None or int_M_results is None:
            n_residual = 0
        else:
            n_residual = len(self.fuse_blocks)
        # print(f"n_residual: {n_residual}")
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for i, up_block in enumerate(self.up_blocks):
                    if i < n_residual:
                        sample = self.fuse_blocks[i](sample, int_M_results[i], w)     
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )

            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for i, up_block in enumerate(self.up_blocks):
                    if i < n_residual:
                        sample = self.fuse_blocks[i](sample, int_M_results[i], w)
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for i, up_block in enumerate(self.up_blocks):
                if i < n_residual:
                    sample = self.fuse_blocks[i](sample, int_M_results[i], w)
                sample = up_block(sample, latent_embeds)
                    
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample