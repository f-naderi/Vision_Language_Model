import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

from .dataset import load_captions, MIN_CAPTION_LEN


# ======================================================================
# Fixed negative captions (for I2T sampled retrieval)
# ======================================================================
def prepare_fixed_captions(all_captions_tuples, test_image_names, seed=42, num_negatives=95, restrict_to_test=False, verbose=True):
    """Pre-generate fixed negative captions per test image."""
    rng = np.random.default_rng(seed)
    test_set = set(test_image_names)
    fixed_negatives = {}

    if restrict_to_test:
        candidate_tuples = [(img, cap) for (img, cap) in all_captions_tuples if img in test_set]
    else:
        candidate_tuples = list(all_captions_tuples)

    # Fast index: image -> captions, and global unique captions (order preserved)
    img_to_caps = defaultdict(list)
    for img, cap in candidate_tuples:
        img_to_caps[img].append(cap)
    all_unique_caps = list(dict.fromkeys(cap for _, cap in candidate_tuples))

    for img_name in test_image_names:
        own_caps = set(img_to_caps.get(img_name, []))
        # list-comprehension preserves order → reproducibility across envs
        pool = [c for c in all_unique_caps if c not in own_caps]

        if len(pool) == 0:
            fixed_negatives[img_name] = []
            continue

        num_neg = min(num_negatives, len(pool))
        idx = rng.choice(len(pool), size=num_neg, replace=False)
        fixed_negatives[img_name] = [pool[i] for i in idx]

    if verbose:
        sizes = [len(v) for v in fixed_negatives.values()]
        if sizes:
            print(
                f"[prepare_fixed_captions] restrict_to_test={restrict_to_test} "
                f"| negatives/img: min={min(sizes)}, max={max(sizes)}, "
                f"mean={sum(sizes)/len(sizes):.1f}"
            )
            if min(sizes) < 50:
                print("[WARN] Some queries have very few negatives — Recall@10 may be inflated.")
    return fixed_negatives


# ======================================================================
# k_values validation
# ======================================================================
def _validate_k_values(k_values, n_candidates, label):
    """Validate k_values against the number of candidates in the retrieval pool."""
    if not k_values:
        raise ValueError(f"{label}: k_values must be non-empty.")
    if any((not isinstance(k, int)) or k < 1 for k in k_values):
        raise ValueError(f"{label}: k_values must contain positive integers, got {k_values}.")
    max_k = max(k_values)
    if n_candidates < max_k:
        raise ValueError(
            f"{label}: pool has {n_candidates} candidates but Recall@{max_k} "
            f"requires at least {max_k}. Reduce k_values or increase negatives."
        )


# ======================================================================
# Image → Text (sampled retrieval)
# ======================================================================
@torch.no_grad()
def evaluate_i2t_recall_at_k(model, test_loader, image_to_captions, fixed_random_captions, processor, device, k_values=(1, 5, 10)):
    """Image-to-Text Retrieval with Recall@K."""
    model.eval()

    if not k_values:
        raise ValueError("k_values must be non-empty.")
    if any((not isinstance(k, int)) or k < 1 for k in k_values):
        raise ValueError(f"k_values must contain positive integers, got {k_values}.")

    recalls = {k: 0 for k in k_values}
    total = 0
    logit_scale = model.get_logit_scale().detach()

    for batch in tqdm(test_loader, desc='Image-to-Text'):
        images = batch['image'].to(device, non_blocking=True)
        image_names = batch['image_name']

        img_feats = model.encode_images(images)  # [B, D]

        for i in range(len(images)):
            img_name = image_names[i]
            gt_captions = list(dict.fromkeys(image_to_captions.get(img_name, [])))
            if not gt_captions:
                raise ValueError(f"No ground-truth captions for {img_name}")

            random_caps = fixed_random_captions.get(img_name, [])
            gt_set = set(gt_captions)
            random_caps = [c for c in random_caps if c not in gt_set]

            all_caps = gt_captions + random_caps
            _validate_k_values(k_values, len(all_caps), f"I2T ({img_name})")

            gt_indices = set(range(len(gt_captions)))

            text_encoding = processor(text=all_caps, return_tensors='pt', padding=True, truncation=True, max_length=77, return_attention_mask=True)

            txt_feats = model.encode_texts(text_encoding['input_ids'].to(device), text_encoding['attention_mask'].to(device))

            similarities = (logit_scale * (img_feats[i:i + 1] @ txt_feats.T)).squeeze(0)

            for k in k_values:
                top_k = similarities.topk(k).indices.tolist()
                if any(idx in gt_indices for idx in top_k):
                    recalls[k] += 1
            total += 1

    if total == 0:
        raise ValueError("Test loader is empty.")

    return {k: recalls[k] / total * 100 for k in k_values}, total


# ======================================================================
# Text → Image (full test set)
# ======================================================================
@torch.no_grad()
def evaluate_t2i_recall_at_k(model, test_loader, image_to_captions, processor, device, k_values=(1, 5, 10)):
    """Text-to-Image Retrieval with Recall@K."""
    model.eval()

    recalls = {k: 0 for k in k_values}
    total = 0

    print("Pre-encoding all test images...")
    all_image_features = []
    all_image_names = []

    for batch in tqdm(test_loader, desc='Encoding images'):
        images = batch['image'].to(device, non_blocking=True)
        img_feats = model.encode_images(images)
        all_image_features.append(img_feats.cpu())
        all_image_names.extend(batch['image_name'])

    if not all_image_features:
        raise ValueError("Test loader is empty.")

    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    print(f"Encoded {len(all_image_names)} images")

    # validate k_values against the actual pool size
    _validate_k_values(k_values, len(all_image_names), "T2I")

    # Efficient duplicate check
    counts = Counter(all_image_names)
    duplicates = [name for name, cnt in counts.items() if cnt > 1]
    if duplicates:
        raise ValueError(f"Duplicate image names: {duplicates[:5]}")

    image_name_to_idx = {name: idx for idx, name in enumerate(all_image_names)}

    # caption -> set of all test image indices that share it
    caption_to_indices = defaultdict(set)
    for img_name in all_image_names:
        idx = image_name_to_idx[img_name]
        for cap in set(image_to_captions.get(img_name, [])):
            caption_to_indices[cap].add(idx)

    logit_scale = model.get_logit_scale().detach()

    for img_name in tqdm(all_image_names, desc='Text-to-Image'):
        gt_captions = list(dict.fromkeys(image_to_captions.get(img_name, [])))
        if not gt_captions:
            raise ValueError(f"No ground-truth captions for {img_name}")

        for query_caption in gt_captions:
            text_encoding = processor(text=[query_caption], return_tensors='pt', padding=True, truncation=True, max_length=77, return_attention_mask=True)
            txt_feat = model.encode_texts(text_encoding['input_ids'].to(device), text_encoding['attention_mask'].to(device))

            similarities = (logit_scale * (txt_feat @ all_image_features.T)).squeeze(0)

            positive_indices = caption_to_indices[query_caption]

            for k in k_values:
                top_k = similarities.topk(k).indices.tolist()
                if any(idx in positive_indices for idx in top_k):
                    recalls[k] += 1
            total += 1

    if total == 0:
        raise ValueError("No queries were evaluated.")

    return {k: recalls[k] / total * 100 for k in k_values}, total