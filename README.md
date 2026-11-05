```markdown
# CLIP Fine-tuning on Flickr8k

A vision-language project that fine-tunes OpenAI's CLIP model on the Flickr8k dataset and evaluates bidirectional retrieval (Image→Text and Text→Image).

## Features

- CLIP ViT-B/32 fine-tuning on Flickr8k
- Bidirectional retrieval: Image→Text and Text→Image
- Outperforms original CLIP by +0.62%
- Only 7.2% of parameters trained (last transformer block of each encoder + projection layers)
- PCA visualization of multimodal embeddings + modality-gap analysis


## Project Structure

Vision_Language_Model/
├── Flickr8k/                      # Flickr8k dataset
├── models/                        # Pre-trained CLIP weights
├── src/                           # Source code
│   ├── dataset.py                 # Data loading pipeline
│   ├── model.py                   # CLIP model wrapper
│   ├── train.py                   # Training utilities
│   ├── evaluate.py                # Retrieval metrics (Recall@K)
│   └── inference.py               # Inference
│   
├── outputs/
│   ├── checkpoints/               # Best model weights
│   ├── inference/                 # Inference output
│   └── results/                   # Visualizations and metrics
│
├── VLM.ipynb                      # Interactive Jupyter notebook
└── requirements.txt               # Python dependencies

> Note: Some files are excluded from the repository due to their large size


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
* Vision Encoder: last transformer block unfrozen
* Text Encoder: last transformer block unfrozen
* Projection Layers: visual_projection and text_projection trainable
* Trainable Parameters: ≈ 10.9M (7.2% of 151.3M)
* Embedding Dimension: 512
* Loss: Symmetric InfoNCE (Cross-Entropy over both directions), averaged
* Optimizer: AdamW with CosineAnnealingLR


## Training Details

* Dataset: Flickr8k (8,091 images)
* Train/Val/Test: 6,472 / 809 / 810
* Image Size: 224×224
* Batch Size: 32
* Epochs 30
* Initial LR: 1e-5 → 1e-6 (cosine decay)
* Weight Decay: 0.01


## Results


Embedding Analysis

* Best Val Loss: 0.1971
* PCA Variance (2D): 10.6% (PC1: 5.4%, PC2: 5.2%)
* Modality Gap: 0.1563
* Avg Pair Distance: 0.9059 ± 0.1038

Retrieval Performance

Image→Text (sampled: 5 GT + 95 negatives = 100 candidates per query, N=810):

| Metric | Score |
|--------|-------|
| Recall@1  | 97.90% |
| Recall@5  | 100.00% |
| Recall@10 | 100.00% |

Text→Image (full test set: 810 candidates per query, N=4049 captions):

| Metric | Score |
|--------|-------|
| Recall@1  | 67.13% |
| Recall@5  | 91.48% |
| Recall@10 | 96.27% |


   
## Requirements

* Python
* PyTorch
* torchvision
* NumPy
* Matplotlib
* scikit-learn
* Transformers
* Pillow
* tqdm
* ipython
* Jupyter Notebook


## References

* Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PmLR.
* Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing image description as a ranking task: Data, models and evaluation metrics. Journal of Artificial Intelligence Research, 47, 853-899.
* Ashish, V. (2017). Attention is all you need. Advances in neural information processing systems.

```