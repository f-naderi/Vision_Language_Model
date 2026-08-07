```markdown
# CLIP Fine-tuning on Flickr8k for Image-Text Retrieval

A vision-language project that fine-tunes OpenAI's CLIP model on the Flickr8k dataset to improve image-to-text retrieval accuracy.

## Features

- CLIP ViT-B/32 fine-tuning on Flickr8k
- Image-to-text retrieval with 98.6% Top-1 accuracy
- Outperforms original CLIP by +1.2%
- Only 6.8% parameters trained (last layer of each encoder)
- PCA visualization of multimodal embeddings


## Project Structure

Vision_Language_Model/
├── Flickr8k/                      # Flickr8k dataset
├── models/                        # Pre-trained CLIP weights
├── src/                           # Source code
│   ├── dataset.py                 # Data loading pipeline
│   ├── model.py                   # CLIP model wrapper
│   ├── train.py                   # Training utilities
│   └── inference.py               # Inference
│   
├── outputs/
│   ├── checkpoints/               # Best model weights
│   └── results/                   # Visualizations and metrics
│
├── VLM.ipynb                      # Interactive Jupyter notebook
└── requirements.txt               # Python dependencies


## Installation

```bash
git clone https://github.com/f-naderi/Vision_Language_Model.git
cd Vision_Language_Model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model Architecture:

* Base Model: CLIP ViT-B/32 (OpenAI)
* Vision Encoder: Last layer unfrozen
* Text Encoder: Last layer unfrozen
* Trainable Parameters: 10.2M (6.8% of 151.3M)
* Embedding Dimension: 512
* Loss Function: Symmetric Cross-Entropy with learned logit_scale
* Optimizer: AdamW with CosineAnnealingLR


## Training Details

* Dataset: Flickr8k (8,091 images)
* Train/Val/Test: 6,472 / 809 / 810
* Image Size: 224×224
* Batch Size: 32
* Epochs 20 (best at epoch 10)
* Initial LR: 1e-5 → 1e-6 (cosine decay)
* Weight Decay: 0.01


## Results

Retrieval Performance (5 GT + 95 random captions)

Metric             Original CLIP    Fine-tuned     (Ours) Improvement
Top-1 Accuracy     97.4%            98.6%           +1.2%
Correct / Total    789 / 810        799 / 810       +10 images

Embedding Analysis

* Best Val Loss: 0.1703
* PCA Variance (2D): ~29.0% (PC1: 22.1%, PC2: 6.8%)
* Modality Gap: 0.796
* Avg Pair Distance: 0.802 ± 0.096

   
## Requirements

* Python
* PyTorch
* NumPy
* Matplotlib
* scikit-learn
* Transformers
* Pillow
* tqdm
* Jupyter Notebook

## References

* Learning Transferable Visual Models From Natural Language Supervision - Radford et al. (2021)
* CLIP: Connecting Text and Images - OpenAI
* Flickr8k Dataset - Hodosh et al.
* Attention Is All You Need - Vaswani et al. (2017)

```