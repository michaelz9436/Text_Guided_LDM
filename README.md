# Text-Conditioned 3D Molecule Generation via E(3)-Equivariant Latent Diffusion

This repository contains the official implementation for **Text-Guided 3D Molecule Generation**. It leverages a fixed-dimensional E(3)-Equivariant Latent Space (see https://github.com/MuZhao2333/MolFLAE, NeurIPS 2025) and introduces a robust Latent Diffusion Model (LDM) conditioned on complex NLP embeddings.

## 🌟 Key Features
- **Fixed-Dimensional Latent Space**: Compresses molecules of arbitrary sizes into exactly 10 nodes ($10 \times 3$ for coordinates, $10 \times 32$ for features), solving the dimension-matching issue in graph diffusion.
- **Strict E(3)-Equivariance**: Uses Cross-Attention to inject text semantic embeddings (384-dim BERT) exclusively into the invariant feature space ($Z_h$), perfectly preserving physical rotational and translational equivariance ($Z_x$).
- **Classifier-Free Guidance (CFG)**: Supports seamless switching between Text-Conditioned and Unconditional generation.


## 📁 Repository Structure
```text
text_guided_LDM/
├── app.py                      # Gradio Web UI and Inference Entry Point
├── checkpoints/                # Pre-trained model weights (Ignored in Git)
├── config/                     # Configuration files for VAE
├── egnn/                       # Equivariant Graph Neural Network implementations
├── equivariant_diffusion/      # Diffusion noise schedules and sampling math
├── lldm/                       # Latent Diffusion Model and Atom Regressor core
├── model/ & utils/             # VAE Decoder (BFN) and Molecule Builder 
└── train/                      # Data pipeline and training scripts
```

## ⚙️ Environment Setup

We highly recommend using `conda` to avoid compilation issues with PyTorch Geometric.

```bash
# 1. Clone the repository
git clone https://github.com/michaelz9436/Text_Guided_LDM.git
cd text_guided_LDM

# 2. Create the conda environment
conda env create -f environment.yaml

# 3. Activate the environment
conda activate tg_LDM

# In case conda would install cpu-only pytorch:
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

## 🚀 Quick Start (Inference & Web UI)

### Download checkpoints

All pretrained checkpoints are hosted on Hugging Face Hub:

https://huggingface.co/Michaelz9436/text-guided-ldm-checkpoints  
You can download them using wget:
```bash
cd checkpoints
wget https://huggingface.co/Michaelz9436/text-guided-ldm-checkpoints/resolve/main/lldm.pt
wget https://huggingface.co/Michaelz9436/text-guided-ldm-checkpoints/resolve/main/regressor.pt
wget https://huggingface.co/Michaelz9436/text-guided-ldm-checkpoints/resolve/main/vae.ckpt
cd ..
```
Or you can use `huggingface_hub`. 

After downloading, your directory should look like:

```text
checkpoints/
├── lldm.pt
├── regressor.pt
└── vae.ckpt
```

### Launch Web UI

Start the interactive interface:

```bash
python app.py
```
> 💡 If you have problem downloading `bert` pretrained weights from hf, you can try hf-mirror: `export HF_ENDPOINT=https://hf-mirror.com`

The application will run at:

```
http://127.0.0.1:7862
```

You can:
- Input natural language prompts
- Adjust CFG guidance scale
- Interactively visualize generated 3D molecules

---

If running on a remote machine, please ensure port forwarding is enabled:

```bash
ssh -L 7862:127.0.0.1:7862 user@your-server
```

Then open in your local browser:

```
http://127.0.0.1:7862
```

## 🏋️‍♂️ Training from Scratch
*This part is not required for inference; it only demonstrates the training pipeline and will not reproduce full training performance as in the provided checkpoints, since the training dataset is not publicly available. See details below.*

Due to the fact that the authors of the original VAE have not yet publicly released the full ZINC9M dataset, and in consideration of data usage permissions and ownership constraints, we are unable to redistribute the complete dataset.

If you require access to the original dataset, please contact the authors of the following repository:
https://github.com/MuZhao2333/MolFLAE

To facilitate reproducibility and testing, we provide a small subset of the original dataset containing 3,000 molecules (data in `train/data/zinc9m_subset`). This subset is intended solely for verifying the full data processing pipeline and training scripts.
We provide a complete, self-contained pipeline for data processing and training (using this subset):

### 1. Data Preparation
Run the data extraction pipeline sequentially. This will extract atom counts, generate rich text descriptions via RDKit, encode them with BERT, and compress molecules into the latent space.

```bash
python train/data_scripts/01_generate_atoms.py
python train/data_scripts/02_generate_desc.py
python train/data_scripts/03_generate_bert.py
python train/data_scripts/04_generate_latent.py
```

### 2. Training
Once the `.pt` latent chunks are generated, you can train the atom regressor and the diffusion model:

```bash
# Train the Atom Count Regressor
python train/train_scripts/train_regressor.py

# Train the Text-Conditioned Latent Diffusion Model
python train/train_scripts/train_diffusion.py
```
*Training configurations can be modified in `train/train_scripts/configs/`.*

## 🙏 Acknowledgements

This project builds upon and is inspired by the following excellent works:

- https://github.com/ehoogeboom/e3_diffusion_for_molecules  
- https://github.com/MuZhao2333/MolFLAE  

We sincerely thank the authors for open-sourcing their code and advancing research in equivariant generative modeling and molecular representation learning.
