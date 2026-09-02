# CamLoop4D: Controllable 4D Scene Generation by Closing the Camera Loop between Generation and Reconstruction
**[Pacific Graphics 2026 — Computer Graphics Forum, Volume 45, Number 7]**

[Hao Li](https://github.com/lihao1477024473)<sup>1</sup>, [Junhao Chen](https://github.com/junhao-c24)<sup>2†</sup>

<sup>1</sup>Independent Researcher, Chengdu, China &nbsp;  <sup>2</sup>Tsinghua University, Beijing, China

†Corresponding author: junhao-c24@mails.tsinghua.edu.cn

## News
- **Pacific Graphics 2026** (Computer Graphics Forum, Vol. 45, No. 7) — camera-ready paper: [📥 **CamLoop4D.pdf**](https://cdn.jsdelivr.net/gh/lihao1477024473/CamLoop4D@main/paper/CamLoop4D.pdf) (main paper + supplementary, direct download). [🔗 GitHub page](https://github.com/lihao1477024473/CamLoop4D/blob/main/paper/CamLoop4D.pdf)

## Method Overview

**Camera-controlled video generation (§3.1).** Camera trajectories are encoded as **Plücker embeddings** $p_{u,v} = [o \times d_{u,v};\ d_{u,v}] \in \mathbb{R}^6$ (with ray direction $d_{u,v} = \mathrm{normalize}(RK^{-1}[u,v,1]^\top)$) and injected into a frozen video diffusion model (AC3D / CogVideoX; generator-agnostic). The trajectory tensor is **directly shared** with reconstruction.

**4D reconstruction with hybrid motion bases (§3.2).** Dynamic scenes are persistent 3D Gaussians whose motion is a linear combination of globally shared bases:

$$T_{0:t} = \exp\!\Big(\sum_{i=1}^{6} \alpha^{(i)}_t G^{(i)}_{fixed} + \sum_{j=1}^{B-6} \beta^{(j)}_t G^{(j)}_{trainable}\Big)$$

- **6 fixed SE(3) generators** (3 translations + 3 rotations, frozen) span the instantaneous motion space (rigid-body priors).
- **Trainable bases** act as a scene-adapted low-rank prior that conditions/regularises optimisation (removing them collapses PSNR by ~5 dB).
- **Motion-coefficient loss** $L_c$ with asymmetric weight $\lambda=0.8$ pushes the optimiser to prefer the rigid generators first.

**Shared trajectory interface (§3.2.3).** Both stages share intrinsics $K$, the world frame at the canonical frame, and the metric scale — removing pose re-estimation (+0.88 dB over DUSt3R poses).

## 1. Video Generation

Generate camera-controlled videos from **text prompts / images** together with **camera trajectories**. The `run_scripts/vgen.sh` script integrates two projects under `vgen/`:

| Method | Repo | Input | Output |
| :--- | :--- | :--- | :--- |
| **AC3D** | `vgen/ac3d-main` | text prompt + camera poses (RealEstate10K format) | camera-controlled video (`CogVideoX` + ControlNet) |
| **SEVA** | `vgen/stable-virtual-camera-main` | single image | video along a preset camera trajectory (`Stable Virtual Camera`) |

### Usage

```bash
bash run_scripts/vgen.sh ac3d-2b    # AC3D: CogVideoX-2b (≈48GB VRAM)
bash run_scripts/vgen.sh ac3d-5b    # AC3D: CogVideoX-5b (≈80GB VRAM)
bash run_scripts/vgen.sh seva       # SEVA: image → preset trajectory video
bash run_scripts/vgen.sh all        # run all of the above sequentially
```

### Configuration

Edit the top of `run_scripts/vgen.sh` to adjust:

- `DATA_ROOT` / `OUT_ROOT` — data and output root paths (defaults to `/root/autodl-tmp/...`)
- `AC3D_DATASET_DIR` / `AC3D_ANNOTATION` — RealEstate10K dataset layout (`annotations/ pose_files/ video_clips/`)
- `AC3D_CKPT_2B` / `AC3D_CKPT_5B` — paths to trained ControlNet checkpoints
- `SEVA_IMG_DIR` — directory of input images (`scene_1.png`, ...)
- `SEVA_TRAJ` — SEVA preset trajectory (`orbit` / `spiral` / `lemniscate` / `zoom-*` / `dolly zoom-*` / `move-*` / `roll`)
- `PROMPT` — text prompt used by AC3D

### Dependencies

```bash
# AC3D
cd vgen/ac3d-main && pip install -r requirements.txt

# SEVA (requires Hugging Face access to `stabilityai/stable-virtual-camera`)
cd vgen/stable-virtual-camera-main && pip install -e .
huggingface-cli login
```

> Note: `vgen.sh` targets a Linux environment (e.g., AutoDL cloud box). SEVA also offers an interactive Gradio UI via `python demo_gr.py`.

## 2. 4D Reconstruction

> Reconstruct a dynamic 4D scene from a single video (or from generated frames). This is the SoM-based reconstruction module with **hybrid motion bases** and the **shared Plücker trajectory interface**.

### Data
Preprocessed nvidia dataset and custom dataset can be found [here](https://drive.google.com/drive/folders/1xzn-Mu_jyr-JTsrERRU-Mh2hQ-NWdfv8). We used [MegaSaM](https://mega-sam.github.io/) to get cameras and depths for custom dataset.

### Installation

The codebase builds on [Shape of Motion](https://github.com/vye16/shape-of-motion) (SoM). To set up:

```
conda create -n som python=3.10
conda activate som
```

Update `requirements.txt` with correct CUDA version for PyTorch and cuUML,
i.e., replacing `cu122` and `cu12` with your CUDA version.

```
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

### Training
To train nvidia dataset
```
python run_training.py \
  --work-dir <OUTPUT_DIR> \
  data:nvidia \
  --data.data-dir </path/to/data>
```

To train custom dataset
```
python run_training.py \
  --work-dir <OUTPUT_DIR> \
  data:custom \
  --data.data-dir </path/to/data>
```

### Train with 2D Gaussian Splatting
To get better scene geometry, we use 2D Gaussian Splatting:

```
python run_training.py \
  --work-dir <OUTPUT_DIR> \
  --use_2dgs
  data:custom \
  --data.data-dir </path/to/data>
```

### Usage

For the generation branch, depth maps (Depth Anything) and 2D tracks (TAPIR) are extracted off-the-shelf from the generated frames; the camera parameters are **not re-estimated** — they come directly from the shared Plücker trajectory. For the reconstruction branch on captured video, we depend on the third-party libraries in `preproc` to generate depth maps, object masks, camera estimates, and 2D tracks.
Please follow the guide in the [preprocessing README](./preproc/README.md).

### Evaluation on iPhone Dataset
First, download our processed iPhone dataset from [this](https://drive.google.com/drive/folders/1xJaFS_3027crk7u36cue7BseAX80abRe?usp=sharing) link. To train on a sequence, e.g., *paper-windmill*, run:

```python
python run_training.py \
  --work-dir <OUTPUT_DIR> \
  --port <PORT> \
  data:iphone \
  --data.data-dir </path/to/paper-windmill/>
```

After optimization, the numerical result can be evaluated via:
```
PYTHONPATH='.' python scripts/evaluate_iphone.py \
  --data_dir </path/to/paper-windmill/> \
  --result_dir <OUTPUT_DIR> \
  --seq_names paper-windmill
```

### Optimization (Section 3.3)
- Adam, lr = 1e-4; 1,000 iterations of initial fitting + 600 epochs of joint optimisation.
- $B = 15$ motion bases (**6 fixed + 9 trainable**), 50,000 initial Gaussians with adaptive density control [KKLD23], 0.5× Gaussian downsampling.
- Overall loss: $L = L_{rgb} + 0.5\,L_{depth} + 0.2\,L_{track} + 0.01\,L_{c}$.

### Results
| Task | Result |
| :--- | :--- |
| **iPhone matched-input** (primary, controlled) | **16.41 dB / 0.621 / 0.440** — +0.52 dB over SoM, +0.19 dB over MoSca (wins 11/14 scenes) |
| **NVIDIA Dynamic Scenes** (second benchmark) | 21.94 dB — +0.20 dB over MoSca, +0.70 dB over SoM (ordering replicates) |
| **System-level (deployment)** | 16.55 dB from generated video, on par with real-video baselines |
| **Geometric self-consistency** (generated scenes) | Camera-following ATE **0.024 m** vs. Free4D cascade 0.112 m (~5× lower) |
| **User study** | 38 participants / 570 judgements; wins over 6 of 7 baselines (Holm–Bonferroni significant) |
| **Ablation** | Hybrid bases +0.64 dB (vs. fully-learnable); fixed-only collapses to 11.52 dB; shared camera +0.88 dB; motion-coeff. loss +0.62 dB |

## 3. Shared Trajectory Interface

> How the camera trajectory flows from the **generation** stage (Section 1) into the **reconstruction** stage (Section 2) without re-estimation. Both stages consume the same trajectory tensor, encoded as Plücker embeddings. A ready-to-use tool is provided: `sharedCamTrajectory.py` (wrapped by `run_scripts/sharedCamTrajectory.sh`).

### Pipeline

```
1. Video Generation (Section 1)
   └─ run_scripts/vgen.sh  →  generated frames + camera trajectory (specified poses)
                         │
                         ▼  (same trajectory, saved on disk)
2. Save trajectory as the reconstructor's camera file
   └─ sharedCamTrajectory.py  →  <out_dir>/droid_recon/<seq_name>.npy
                                 <out_dir>/camera/<seq>_<i>.json   (iPhone style)
                         │
                         ▼  (loaded directly, NO pose re-estimation)
3. 4D Reconstruction (Section 2)
   └─ python run_training.py data:custom \
        --data.data-dir <out_dir> \
        --data.camera-type droid_recon \
        --data.seq_name <seq_name>
```

### Tools

`sharedCamTrajectory.py` (requires a Python env with `numpy`, e.g. `miniconda3`):

```bash
# 1) Generate a camera trajectory (+ Plücker embedding) and save the shared file
python sharedCamTrajectory.py plucker-trajectory \
    --traj-type spiral --num-frames 80 \
    --fx 700 --fy 700 --cx 360 --cy 240 --height 480 --width 720 \
    --out-dir ./data --seq-name my_scene

# 2) Convert an existing trajectory (c2w .npy) into the reconstructor camera file
python sharedCamTrajectory.py convert \
    --traj-c2w traj_c2w.npy --intrinsics "700,700,360,240" --img-shape "480,720" \
    --out-dir ./data --seq-name my_scene

# 3) Convert a RealEstate10K pose file into iPhone-style camera JSONs
python sharedCamTrajectory.py realestate10k-to-iphone \
    --pose-file pose_files/0000cc6d8b108390.txt \
    --out-dir ./data --seq-name my_scene

# 4) Inspect a shared camera file
python sharedCamTrajectory.py inspect --cam-file ./data/droid_recon/my_scene.npy
```

A shell wrapper (`run_scripts/sharedCamTrajectory.sh`) with the same sub-commands and AutoDL-style configurable paths is also available:

```bash
bash run_scripts/sharedCamTrajectory.sh plucker-trajectory        # or convert / realestate10k-to-iphone / inspect
# pass options via env vars, e.g.:
SEQ_NAME=my_scene OUT_DIR=/root/autodl-tmp/data/seva2som \
PYTHON="C:/Users/Administrator/miniconda3/python.exe" \
bash run_scripts/sharedCamTrajectory.sh realestate10k-to-iphone
```

### Notes
- Camera sources: `droid_recon` (loads `droid_recon/<seq_name>.npy`) or `megasam` (loads `<seq_name>.npz`).
- `realestate10k-to-iphone` writes both the iPhone-style `camera/*.json` (for `iPhoneDataset camera_type="original"`) and the `droid_recon/<seq>.npy` shared file.
- The interface removes the *pose-estimation stage* (residual generator deviation: ATE ≈ 0.021 m, angular 1.3°), yielding **+0.88 dB PSNR** over DUSt3R-estimated poses.

## Citation
If you find this project useful for your research, please cite:

```
@article{camloop4d2026,
  title   = {CamLoop4D: Controllable 4D Scene Generation by Closing the Camera Loop between Generation and Reconstruction},
  author  = {Li, Hao and Chen, Junhao},
  journal = {Computer Graphics Forum (Pacific Graphics 2026)},
  volume  = {45},
  number  = {7},
  year    = {2026}
}
```

The reconstruction stage builds upon (base method):
```
@inproceedings{som2024,
  title     = {Shape of Motion: 4D Reconstruction from a Single Video},
  author    = {Wang, Qianqian and Ye, Vickie and Gao, Hang and Zeng, Weijia and Austin, Jake and Li, Zhengqi and Kanazawa, Angjoo},
  journal   = {arXiv preprint arXiv:2407.13764},
  year      = {2024}
}
```
