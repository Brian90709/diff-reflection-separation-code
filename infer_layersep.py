# Last modified: 2024-05-24
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
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
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
"""Layer-separation reflection removal inference.

Runs the Marigold-derived LayersepPipeline over one or more reflection
benchmark datasets, writing the transmission layer, reflection layer,
and pre-optimization transmission for every input frame.
"""

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn import Conv2d
from torch.nn.parameter import Parameter

from marigold.layersep_pipeline import LayersepPipeline
from src.util.seeding import seed_all

# --------------------------------------------------------------------------
# Paths & dataset registry
# --------------------------------------------------------------------------

# Stable Diffusion 2 base model. The original `stabilityai/stable-diffusion-2`
# repo was pulled from the Hub, so we default to an architecturally identical
# community re-upload (same VAE / OpenCLIP text encoder / v-prediction scheduler).
# Override with --base_model if you have your own SD-2 (768, v-prediction) repo.
SD_BASE_MODEL = "sd2-community/stable-diffusion-2-1"

CHECKPOINT_ROOT = Path("./checkpoints")
UNET_CHECKPOINT = CHECKPOINT_ROOT / "iter_016000"
REFINER_CHECKPOINT = CHECKPOINT_ROOT / "fuse_blocks.bin"
LRM_CHECKPOINT = CHECKPOINT_ROOT / "lrm/iter_008000/aux_net.bin"

DEFAULT_OUTPUT_ROOT = Path("./inference_ori")

# Absolute paths to the reflection-removal test datasets on this cluster.
# Adjust DATA_ROOT when moving the release to a different machine.
DATA_ROOT = Path("/inspire/hdd/global_user/zhangkaipeng-24043/hzh")
DATASET_GLOBS = {
    "real20_420":         DATA_ROOT / "reflection-removal/test/real20_420/blended/*",
    "Nature":             DATA_ROOT / "reflection-removal/test/Nature/blended/*",
    "PostcardDataset":    DATA_ROOT / "reflection-removal/test/SIR2/PostcardDataset/blended/*",
    "SolidObjectDataset": DATA_ROOT / "reflection-removal/test/SIR2/SolidObjectDataset/blended/*",
    "WildSceneDataset":   DATA_ROOT / "reflection-removal/test/SIR2/WildSceneDataset/blended/*",
    "real45":             DATA_ROOT / "reflection-removal/test/real45/*",
    "NTIRE25":            DATA_ROOT / "NTIRE2025_Challenge_SIRR/val_100/blended/*",
    "selected":           DATA_ROOT / "NTIRE2025_Challenge_SIRR/selected/*",
    "Mine5":              DATA_ROOT / "Mine5/*",
    "liu":                DATA_ROOT / "Liu_reflection-removal/reflection/Huang and Liu_input/*",
    "xue":                DATA_ROOT / "Liu_reflection-removal/reflection/Xue et al_input/*",
    "kopf":               DATA_ROOT / "Liu_reflection-removal/reflection/Kopf et al_input/*",
}

DEFAULT_TEST_DATASETS = [
    "real20_420", "Nature", "PostcardDataset", "SolidObjectDataset", "WildSceneDataset",
]


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------

def _replace_unet_conv_in(model):
    """Widen the UNet's first conv from 4 to 8 input channels.

    Duplicates the original 4-channel weight along the input-channel
    dimension and halves its magnitude, so the widened conv outputs
    the same activations when fed two stacked copies of the original
    latent. Same trick used in the Marigold paper.
    """
    weight = model.unet.conv_in.weight.clone()         # [320, 4, 3, 3]
    bias = model.unet.conv_in.bias.clone()             # [320]
    weight = weight.repeat((1, 2, 1, 1)) * 0.5         # [320, 8, 3, 3]

    new_conv = Conv2d(
        8,
        model.unet.conv_in.out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    )
    new_conv.weight = Parameter(weight)
    new_conv.bias = Parameter(bias)
    model.unet.conv_in = new_conv
    model.unet.config["in_channels"] = 8
    return model


def build_pipeline(checkpoint_dir, dtype, variant, device, base_model=SD_BASE_MODEL):
    """Load SD-2, widen its UNet, and overlay trained layer-separation weights."""
    pipe = LayersepPipeline.from_pretrained(
        base_model, variant=variant, torch_dtype=dtype,
    )
    pipe = _replace_unet_conv_in(pipe)

    pipe._set_class_embedding()
    pipe.set_decoder()

    unet_weights = os.path.join(str(checkpoint_dir), "unet", "diffusion_pytorch_model.bin")
    pipe.unet.load_state_dict(torch.load(unet_weights, map_location=device))
    pipe.unet.to(device)

    pipe._load_refiner_checkpoints(str(REFINER_CHECKPOINT), device)
    pipe._load_lrm_checkpoints(str(LRM_CHECKPOINT), device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("running without xformers")

    return pipe.to(device)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run layer-separation reflection removal (LayersepPipeline)."
    )

    # Model / checkpoint
    parser.add_argument("--checkpoint", type=str, default=str(UNET_CHECKPOINT),
                        help="Directory containing unet/diffusion_pytorch_model.bin")
    parser.add_argument("--base_model", type=str, default=SD_BASE_MODEL,
                        help="SD-2 base model repo/path (768, v-prediction) "
                             "supplying the VAE, text encoder, tokenizer and scheduler.")

    # Inference
    parser.add_argument("--denoise_steps", type=int, default=50,
                        help="Diffusion denoising steps (quant. eval uses 50).")
    parser.add_argument("--ensemble_size", type=int, default=1,
                        help="Number of predictions to ensemble.")
    parser.add_argument("--half_precision", "--fp16", action="store_true",
                        help="Run with fp16 (faster, slight accuracy loss).")

    # Resolution
    parser.add_argument("--processing_res", type=int, default=0,
                        help="Resize input to at most this; 0 means keep native resolution.")
    parser.add_argument("--resolution", type=int, default=960,
                        help="Operating resolution for the inference pass.")
    parser.add_argument("--output_processing_res", action="store_true",
                        help="Emit result at processing resolution instead of input resolution.")
    parser.add_argument("--resample_method", type=str, default="bilinear",
                        choices=["bilinear", "nearest"])

    # Ablation switches (paper defaults: all on, w = 0.8)
    parser.add_argument("--optimization", action=argparse.BooleanOptionalAction, default=True,
                        help="Latent optimization (on by default; --no-optimization to disable).")
    parser.add_argument("--s_sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="Disjoint/separation sampling (on by default; --no-s_sampling to disable).")
    parser.add_argument("--w", type=float, default=0.8,
                        help="FGFM/CFW refiner blend weight for the transmission "
                             "(paper default 0.8; 0 = no refinement).")

    # I/O
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Folder of input images to process (jpg/png/bmp/webp). "
                             "When set, the built-in dataset registry is ignored.")
    parser.add_argument("--save_to_dir", type=str, default=None,
                        help=f"Output root. Default: {DEFAULT_OUTPUT_ROOT}/{{resolution}}_op/{{ds}}")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_TEST_DATASETS,
                        choices=list(DATASET_GLOBS.keys()),
                        help="Which test sets to evaluate (space-separated). "
                             "Ignored when --input_dir is given.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()


# --------------------------------------------------------------------------
# Per-dataset loop
# --------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _collect_input_frames(input_dir):
    """Return a sorted list of image paths directly under `input_dir`."""
    frames = []
    for ext in IMAGE_EXTENSIONS:
        frames.extend(glob.glob(str(Path(input_dir) / f"*{ext}")))
        frames.extend(glob.glob(str(Path(input_dir) / f"*{ext.upper()}")))
    return sorted(set(frames))


def run_dataset(pipe, ds, args, generator):
    frames = glob.glob(str(DATASET_GLOBS[ds]))

    if args.save_to_dir is not None:
        out_dir = Path(args.save_to_dir) / ds
    else:
        out_dir = DEFAULT_OUTPUT_ROOT / f"{args.resolution}_op" / ds

    run_frames(pipe, frames, out_dir, args, generator)


def run_frames(pipe, frames, out_dir, args, generator):
    """Run the pipeline over `frames`, writing three layers per input to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    match_input_res = not args.output_processing_res

    for frame in frames:
        print(frame)
        stem = Path(frame).stem
        transmission_path = out_dir / f"{stem}_transmission.png"
        if transmission_path.exists():
            continue

        input_image = Image.open(frame)
        pipe_out = pipe(
            input_image,
            denoising_steps=args.denoise_steps,
            ensemble_size=args.ensemble_size,
            processing_res=args.processing_res,
            match_input_res=match_input_res,
            batch_size=1,
            show_progress_bar=False,
            resample_method=args.resample_method,
            cross_attention=True,
            generator=generator,
            inf_res=args.resolution,
            w=args.w,
            optimization=args.optimization,
            s_sampling=args.s_sampling,
        )

        pipe_out.transmission_Image.save(transmission_path)
        pipe_out.reflection_Image.save(out_dir / f"{stem}_reflection.png")
        pipe_out.ori_transmission_Image.save(out_dir / f"{stem}_ori_transmission.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    print(f"arguments: {args}")

    if args.processing_res == 0 and args.output_processing_res:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to "
            "exactly the same resolution, due to padding/pooling properties of conv layers."
        )
    if args.ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")

    seed = args.seed if args.seed is not None else int(time.time())
    seed_all(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    if args.half_precision:
        dtype, variant = torch.float16, "fp16"
        logging.warning(f"Running with half precision ({dtype}); results may be suboptimal.")
    else:
        dtype, variant = torch.float32, None

    logging.info(
        f"Inference settings: checkpoint=`{args.checkpoint}`, "
        f"denoise_steps={args.denoise_steps}, ensemble_size={args.ensemble_size}, "
        f"processing_res={args.processing_res}, seed={seed}"
    )

    pipe = build_pipeline(Path(args.checkpoint), dtype, variant, device, base_model=args.base_model)
    generator = torch.Generator(device=device).manual_seed(2024)

    with torch.no_grad():
        if args.input_dir is not None:
            frames = _collect_input_frames(args.input_dir)
            if not frames:
                logging.warning(f"No images found under {args.input_dir}")
            if args.save_to_dir is not None:
                out_dir = Path(args.save_to_dir)
            else:
                out_dir = DEFAULT_OUTPUT_ROOT / Path(args.input_dir).name
            run_frames(pipe, frames, out_dir, args, generator)
        else:
            for ds in args.datasets:
                run_dataset(pipe, ds, args, generator)


if __name__ == "__main__":
    main()
