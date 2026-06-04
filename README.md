<div align="center">

<h2><img src="logo.svg" height="26" align="absmiddle" alt="DiffRS logo"> Reflection Separation from a Single Image via Joint Latent Diffusion</h2>

[![project page](https://img.shields.io/badge/Project-Page-2ea44f)](https://brian90709.github.io/diff-reflection-separation/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)&nbsp;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Brian9999/diff-reflection-separation)&nbsp;

[Zheng-Hui Huang](https://brian90709.github.io/)<sup>1,2</sup>, [Zhixiang Wang](https://lightchaserx.github.io)<sup>1&#42;</sup>, [Yu-Lun Liu](https://yulunalexliu.github.io)<sup>3</sup>, [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)<sup>2</sup>

<sup>1</sup>Shanda AI Research Tokyo &nbsp; <sup>2</sup>National Taiwan University &nbsp; <sup>3</sup>National Yang Ming Chiao Tung University

<sup>&#42;</sup>Corresponding author

</div>

---

Separate a single photo taken through glass into a **transmission** layer (the
reflection-free image) and a **reflection** layer, using a Stable Diffusion 2
model fine-tuned to generate both at once. See the
[project page](https://brian90709.github.io/diff-reflection-separation/)
for the method.

## 🛠️ Install

```bash
conda create -y -n diffrs python=3.10 && conda activate diffrs
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the pre-trained weights into `./checkpoints`:

```bash
huggingface-cli download Brian9999/diff-reflection-separation --repo-type model --local-dir ./checkpoints
```

The SD-2 base model is fetched automatically on first run (default
`sd2-community/stable-diffusion-2-1`; override with `--base_model`).

## 🚀 Run

```bash
python infer_layersep.py --input_dir ./samples --save_to_dir ./outputs
```

The defaults match the paper setting (960×960, `w = 0.8`, disjoint sampling
`k = 0.2`, latent optimization on). Point `--input_dir` at any folder of images.
Each input yields three files: `*_transmission.png` (the result),
`*_reflection.png`, and `*_ori_transmission.png` (transmission before CFW
refinement). Run with `-h` for all options.

### 🎛️ Ablation switches

All are **on by default**; disable individually to study their effect.

| Flag | Effect |
| --- | --- |
| `--no-optimization` | Turn off latent optimization (learned composition module, LRM). |
| `--no-s_sampling` | Turn off disjoint sampling (strength `k = 0.2`). |
| `--w <float>` | FGFM/CFW refiner strength on the transmission (default `0.8`; `0` = off). |

Latent optimization is used in the paper and is on by default. For faster
inference, turn it off with `--no-optimization` (at a small quality trade-off).

## 🗂️ Data

We use the same training and test data as
[DSRNet](https://github.com/mingcv/DSRNet). Please refer to their repository for
dataset preparation and download links.

## 📑 Citation

```bibtex
@inproceedings{huang2026reflection,
  title     = {Reflection Separation from a Single Image via Joint Latent Diffusion},
  author    = {Huang, Zheng-Hui and Wang, Zhixiang and Liu, Yu-Lun and Chuang, Yung-Yu},
  booktitle = {CVPR},
  year      = {2026}
}
```