import os
import random
from collections import defaultdict, Counter

import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from .dataset import load_captions, get_image_path_or_raise


# ======================================================================
# Helpers
# ======================================================================
def _ensure_model_on_device(model, device):
    """Move model to device and set eval mode."""
    model.to(device)
    model.eval()
    return model


def _load_best_model(model, device, checkpoint_path='outputs/checkpoints/best_model.pth'):
    """Load the best-model checkpoint into `model`."""
    print("Loading best model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = _ensure_model_on_device(model, device)
    print(f"Loaded from epoch {checkpoint['epoch'] + 1}, Val loss: {checkpoint['val_loss']:.4f}")
    return model


def _resolve_correct_images(all_captions_tuples, query_caption, test_image_set):
    """Return all test images (unique, order-preserved) that share `query_caption`."""
    return list(dict.fromkeys(
        img for img, cap in all_captions_tuples
        if cap == query_caption and img in test_image_set
    ))


# ======================================================================
# Image-to-Text retrieval
# ======================================================================
def run_inference(model, test_loader, IMAGE_DIR, CAPTIONS_FILE, device='cuda', output_dir='outputs/inference'):
    model = _load_best_model(model, device)

    os.makedirs(output_dir, exist_ok=True)

    image_to_captions, all_captions_tuples = load_captions(CAPTIONS_FILE)

    dataset = test_loader.dataset
    if len(dataset) == 0:
        raise ValueError("Test dataset is empty.")

    rng = random.Random(42)
    random_idx = rng.randint(0, len(dataset) - 1)
    sample = dataset[random_idx]

    sample_image_name = sample['image_name']
    sample_image_tensor = sample['image']
    sample_image_path = get_image_path_or_raise(IMAGE_DIR, sample_image_name)

    # Unique ground-truth captions for this image
    ground_truth_captions = list(dict.fromkeys(image_to_captions.get(sample_image_name, [])))
    if not ground_truth_captions:
        raise ValueError(f"No ground-truth captions for '{sample_image_name}'")

    print(f"\n{'-' * 60}")
    print(f"Test Image: {sample_image_name}")
    print(f"{'-' * 60}")
    print("Ground Truth Captions:")
    for gt in ground_truth_captions:
        print(f"  • {gt}")

    with Image.open(sample_image_path) as img:
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Test Image', fontsize=14, fontweight='bold')
        fig.savefig(os.path.join(output_dir, 'i2t_query_image.png'), bbox_inches='tight')
        plt.close(fig)

    # Deduplicate negatives BY CAPTION TEXT, excluding the query's own GT
    gt_set = set(ground_truth_captions)
    other_captions = list(dict.fromkeys(
        c for n, c in all_captions_tuples
        if n != sample_image_name and c not in gt_set
    ))

    num_random = min(95, len(other_captions))
    random_captions = rng.sample(other_captions, num_random)

    caption_texts = ground_truth_captions + random_captions
    gt_indices = list(range(len(ground_truth_captions)))

    print(f"\nSearching among {len(caption_texts)} captions (from full dataset)...")

    processor = model.get_preprocess()
    image_tensor = sample_image_tensor.unsqueeze(0).to(device)

    text_encoding = processor(text=caption_texts, return_tensors='pt', padding=True, truncation=True, max_length=77, return_attention_mask=True)

    img_feat = model.encode_images(image_tensor)
    txt_feat = model.encode_texts(text_encoding['input_ids'].to(device), text_encoding['attention_mask'].to(device))

    logit_scale = model.get_logit_scale().detach()
    similarities = (logit_scale * (img_feat @ txt_feat.T)).squeeze(0)

    # Relative softmax over the chosen pool — NOT a calibrated confidence
    probabilities = torch.softmax(similarities, dim=0)

    top5 = similarities.topk(min(5, len(caption_texts)))

    print(f"\n{'-' * 60}")
    print("Top-5 Retrieved Captions:")
    print(f"{'-' * 60}")
    for rank, (idx_pred, score) in enumerate(zip(top5.indices, top5.values)):
        idx_pred = idx_pred.item()
        caption = caption_texts[idx_pred]
        prob = probabilities[idx_pred].item()
        is_gt = " GROUND TRUTH" if idx_pred in gt_indices else ""
        print(f"  {rank + 1}. [{score:.1f}] ({prob:.1%}) {caption[:100]} {is_gt}")

    top5_idx = [i.item() for i in top5.indices]
    found_gt = sum(1 for g in gt_indices if g in top5_idx)

    print(f"\n{'-' * 60}")
    print(f"Top-5 Ground Truth Retrieval: {found_gt}/{len(ground_truth_captions)}")
    print(f"  (in candidate pool of {len(caption_texts)} captions)")
    print(f"  Best score: {similarities[top5.indices[0]].item():.1f}")
    print(f"  logit_scale: {logit_scale.item():.3f}")
    print(f"{'-' * 60}")

    return top5, caption_texts, ground_truth_captions


# ======================================================================
# Text-to-Image retrieval
# ======================================================================
def run_t2i_inference(model, test_loader, IMAGE_DIR, CAPTIONS_FILE, device='cuda', query_caption=None, output_dir='outputs/inference'):
    model = _load_best_model(model, device)

    os.makedirs(output_dir, exist_ok=True)

    image_to_captions, all_captions_tuples = load_captions(CAPTIONS_FILE)

    print("Encoding all test images...")
    all_image_features = []
    all_image_names = []

    for batch in test_loader:
        images = batch['image'].to(device, non_blocking=True)
        img_feats = model.encode_images(images)
        all_image_features.append(img_feats.cpu())
        all_image_names.extend(batch['image_name'])

    if not all_image_features:
        raise ValueError("No images encoded — test loader produced no batches.")

    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    print(f"Encoded {len(all_image_names)} images")

    # Duplicate check on actually encoded image names
    counts = Counter(all_image_names)
    duplicates = [name for name, cnt in counts.items() if cnt > 1]
    if duplicates:
        raise ValueError(f"Duplicate image names in test loader: {duplicates[:5]}")

    # Use actually-encoded image set (not dataset.images)
    test_image_set = set(all_image_names)

    rng = random.Random(42)

    if query_caption is None:
        # Pick from actually-encoded names
        random_idx = rng.randint(0, len(all_image_names) - 1)
        correct_image_name = all_image_names[random_idx]
        gt_caps = list(dict.fromkeys(image_to_captions.get(correct_image_name, [])))
        if not gt_caps:
            raise ValueError(f"No ground-truth captions for '{correct_image_name}'")
        query_caption = rng.choice(gt_caps)
        possible_correct_images = _resolve_correct_images(all_captions_tuples, query_caption, test_image_set)
        if not possible_correct_images:
            print(f"[WARN] Caption '{query_caption}' not found in all_captions_tuples; "
                  f"falling back to '{correct_image_name}'.")
            possible_correct_images = [correct_image_name]
    else:
        possible_correct_images = _resolve_correct_images(all_captions_tuples, query_caption, test_image_set)
        if not possible_correct_images:
            print("\nQuery caption not found in test set!")
            print(f'  Query: "{query_caption}"')
            return None, None, None

    print(f"\n{'-' * 60}")
    print(f'Query Caption: "{query_caption}"')
    print(f"  Ground Truth Image(s): {len(possible_correct_images)}")
    for name in possible_correct_images[:5]:
        print(f"     • {name}")
    print(f"{'-' * 60}")

    processor = model.get_preprocess()
    text_encoding = processor(text=[query_caption], return_tensors='pt', padding=True, truncation=True, max_length=77, return_attention_mask=True)

    query_txt_feat = model.encode_texts(text_encoding['input_ids'].to(device), text_encoding['attention_mask'].to(device))

    logit_scale = model.get_logit_scale().detach()
    similarities = (logit_scale * (query_txt_feat @ all_image_features.T)).squeeze(0)

    top5 = similarities.topk(min(5, len(all_image_names)))

    num_results = len(top5.indices)
    if num_results == 0:
        raise ValueError("No results to visualize.")

    fig, axes = plt.subplots(1, num_results, figsize=(4 * num_results, 4))
    if num_results == 1:
        axes = [axes]

    for rank, (idx, score) in enumerate(zip(top5.indices, top5.values)):
        img_name = all_image_names[idx.item()]
        try:
            img_path = get_image_path_or_raise(IMAGE_DIR, img_name)
            with Image.open(img_path) as img:
                axes[rank].imshow(img)
        except Exception:
            axes[rank].text(0.5, 0.5, f"Error:\n{img_name}", ha='center')

        axes[rank].axis('off')
        tag = " correct" if img_name in possible_correct_images else ""
        axes[rank].set_title(f"Top-{rank + 1}\n[{score:.1f}] {tag}", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 't2i_top5.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Top-5 results saved to: {save_path}")

    top5_names = [all_image_names[i.item()] for i in top5.indices]
    correct_found = [n for n in possible_correct_images if n in top5_names]

    print(f"\n{'-' * 60}")
    print("Top-5 Image Retrieval:")
    print(f"{'-' * 60}")
    print(f'   Query: "{query_caption}"')
    print(f"   Ground truth image(s): {len(possible_correct_images)}")

    if correct_found:
        rank = top5_names.index(correct_found[0]) + 1
        print(f" Correct image found at rank {rank}")
    else:
        print(" Correct image NOT in top-5")
    print(f"{'-' * 60}")

    return query_caption, possible_correct_images, top5_names