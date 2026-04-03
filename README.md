# 🧠 Image Compression & Reconstruction via Autoencoders

> Deep learning pipeline for image compression using Artificial Neural Networks and Convolutional Neural Networks — built from scratch in PyTorch.

-----

## 📌 Overview

An **autoencoder** learns to compress images into a compact latent representation and reconstruct them back. This project implements and compares **4 architectures** across two backbone families, exploring how bottleneck size and depth affect reconstruction quality.

```
Input Image → [ Encoder ] → Latent Vector (32 or 64-dim) → [ Decoder ] → Reconstructed Image
```

**Dataset:** CIFAR-10 — 60,000 RGB images across 10 classes (32×32px)  
**Framework:** PyTorch (built from scratch, no pretrained models)

-----

## 🏗️ Model Architectures

### ANN-v1 — Shallow Fully Connected

|Layer            |Dims                     |
|-----------------|-------------------------|
|Input (flattened)|3072                     |
|Encoder          |3072 → 512 → 128 → **32**|
|Decoder          |32 → 128 → 512 → 3072    |

- 3-layer symmetric encoder-decoder
- ReLU activations, Tanh output
- Bottleneck: **32** → Compression ratio: **96×**

-----

### ANN-v2 — Deep Fully Connected + Regularised

|Layer            |Dims                                  |
|-----------------|--------------------------------------|
|Input (flattened)|3072                                  |
|Encoder          |3072 → 1024 → 512 → 256 → 128 → **64**|
|Decoder          |64 → 128 → 256 → 512 → 1024 → 3072    |

- 5-layer deep encoder-decoder
- BatchNorm + LeakyReLU + Dropout (0.2) for regularisation
- Bottleneck: **64** → Compression ratio: **48×**

-----

### CNN-v1 — Standard Convolutional

```
Encoder: 32×32×3 → Conv(32) → Conv(64) → Conv(128) → FC(32)
Decoder: FC(2048) → ConvT(64) → ConvT(32) → ConvT(3) → 32×32×3
```

- Stride-2 convolutions for spatial downsampling
- BatchNorm after every conv layer
- ConvTranspose2d for upsampling
- Bottleneck: **32** → Compression ratio: **96×**

-----

### CNN-v2 — Deep Convolutional + Residual Blocks ⭐

```
Encoder: 32×32 → [Conv+ResBlock]×4 → 2×2×512 → FC(64)
Decoder: FC → 2×2×512 → [ResBlock+ConvT]×4 → 32×32×3
```

- 4 downsampling + 4 upsampling stages
- **Residual (skip) connections** preserve gradient flow
- Deepest and most expressive architecture
- Bottleneck: **64** → Compression ratio: **48×**

-----

## 📊 Results

|Model |Type    |Bottleneck|Params|Test MSE|PSNR|
|------|--------|----------|------|--------|----|
|ANN-v1|FC      |32        |~2M   |—       |—   |
|ANN-v2|FC+BN   |64        |~10M  |—       |—   |
|CNN-v1|Conv    |32        |~1M   |—       |—   |
|CNN-v2|Conv+Res|64        |~5M   |—       |—   |


> 📝 Fill in MSE and PSNR values from Cell 16 output after training.

**Key finding:** CNN-v2 achieves the best reconstruction quality (highest PSNR) because convolutional layers exploit spatial structure in images — something fully connected layers fundamentally cannot do. Residual connections allow deeper training without vanishing gradients.

-----

## 🗂️ Project Structure

```
autoencoder-project/
├── autoencoder_premium.ipynb   # Main notebook (all code + outputs)
├── loss_curves.png             # Train/val loss for all 4 models
├── reconstructions.png         # Original vs reconstructed comparison
├── latent_pca.png              # PCA projection of latent space
├── latent_tsne_cnn2.png        # t-SNE visualisation (CNN-v2)
├── summary_report.png          # Grand summary figure
├── compression_ratio.png       # Compression ratio bar chart
├── interp_cnn2.png             # Latent space interpolation (CNN-v2)
├── interp_ann2.png             # Latent space interpolation (ANN-v2)
├── ANN-v1_BN32.pth             # Saved model weights
├── ANN-v2_BN64.pth
├── CNN-v1_BN32.pth
└── CNN-v2_BN64.pth
```

-----

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/autoencoder-project
cd autoencoder-project

# 2. Install dependencies
pip3 install torch torchvision matplotlib scikit-learn tqdm pillow notebook

# 3. Launch notebook
jupyter notebook autoencoder_premium.ipynb

# 4. Run all cells top to bottom (Shift+Enter)
#    Training takes ~30-60 min on CPU
```

-----

## 📈 Visualisations

- **Loss curves** — Training & validation MSE (log scale) for all 4 models
- **Reconstruction grid** — Original vs reconstructed images per model
- **Latent space PCA** — 2D projection of 32/64-dim latent vectors
- **t-SNE** — Non-linear latent space clustering (CNN-v2)
- **Interpolation** — Smooth traversal between two images through latent space

-----

## 🛠️ Built With

- PyTorch 2.11
- torchvision
- scikit-learn (PCA, t-SNE)
- matplotlib

-----

*BUILT FRO AIMS*
