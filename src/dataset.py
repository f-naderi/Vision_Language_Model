import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import csv
import random
from collections import defaultdict
from functools import partial
from torchvision import transforms as T


MIN_CAPTION_LEN = 5


def load_captions(captions_file, min_caption_len=MIN_CAPTION_LEN, verbose=True):
    """Load captions from a CSV/TSV file, filtering out short ones."""
    image_to_captions = defaultdict(list)
    all_captions_tuples = []

    with open(captions_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)

        delimiter = '\t' if '\t' in first_line else ','
        reader = csv.reader(f, delimiter=delimiter)

        try:
            first_row = next(reader)
        except StopIteration:
            raise ValueError(f"Captions file is empty: {captions_file}")

        is_header = (
            first_row
            and first_row[0].lower() in ['image', 'image_name', 'img', 'filename']
        )

        # Process the first row if there is no header
        if not is_header and first_row and len(first_row) >= 2:
            img_name = first_row[0].strip()
            caption = first_row[1].strip()
            if img_name and caption and len(caption) >= min_caption_len:
                image_to_captions[img_name].append(caption)
                all_captions_tuples.append((img_name, caption))

        # Process the remaining rows
        for row in reader:
            if len(row) >= 2:
                img_name = row[0].strip()
                caption = row[1].strip()
                if img_name and caption and len(caption) >= min_caption_len:
                    image_to_captions[img_name].append(caption)
                    all_captions_tuples.append((img_name, caption))

    # Remove images that ended up with no valid captions
    removed = 0
    for img_name in list(image_to_captions.keys()):
        if not image_to_captions[img_name]:
            del image_to_captions[img_name]
            removed += 1

    if verbose:
        print(f"[load_captions] {len(image_to_captions)} images, "
              f"{len(all_captions_tuples)} captions "
              f"(min_caption_len={min_caption_len})")
        if removed > 0:
            print(f"[load_captions] Removed {removed} images with no valid captions")

    return dict(image_to_captions), all_captions_tuples


def resolve_image_path(image_dir, img_name):
    """Return the image path if it exists (trying common extensions), else None."""
    img_path = os.path.join(image_dir, img_name)
    if os.path.exists(img_path):
        return img_path
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        test_path = img_path + ext
        if os.path.exists(test_path):
            return test_path
    return None


def get_image_path_or_raise(image_dir, img_name):
    """Same as resolve_image_path, but raises FileNotFoundError if missing."""
    path = resolve_image_path(image_dir, img_name)
    if path is None:
        raise FileNotFoundError(f"Image not found: {img_name}")
    return path


# DataLoader worker init
def _worker_init_fn(worker_id):
    """Give each DataLoader worker its own RNG seed."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    dataset._rng = random.Random(dataset._seed + worker_id * 1009)
    # Also seed torch's RNG per worker (important for torchvision transforms)
    torch.manual_seed(dataset._seed + worker_id * 1009)


def build_train_transform(image_size=224):
    """Build the augmentation pipeline for the training split."""
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.11)),
        T.RandomHorizontalFlip(p=0.5)
    ])


# Dataset
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, processor, split='train', seed=42,
                 filter_missing=True, min_caption_len=MIN_CAPTION_LEN,
                 augment=True, image_size=224):
        self.image_dir = image_dir
        self.processor = processor
        self.split = split
        self.filter_missing = filter_missing

        # Load captions via shared function
        self.image_captions, _ = load_captions(captions_file, min_caption_len=min_caption_len, verbose=False)

        # Split
        unique_images = list(self.image_captions.keys())
        total = len(unique_images)

        if total == 0:
            raise ValueError("No images found after filtering!")

        rng = random.Random(seed)
        rng.shuffle(unique_images)

        if split == 'train':
            self.images = unique_images[:int(0.8 * total)]
        elif split == 'val':
            self.images = unique_images[int(0.8 * total):int(0.9 * total)]
        elif split == 'test':
            self.images = unique_images[int(0.9 * total):]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")

        # Filter missing image files
        if filter_missing:
            original_len = len(self.images)
            self.images = [
                im for im in self.images
                if resolve_image_path(self.image_dir, im) is not None
            ]
            if len(self.images) < original_len:
                print(f"[WARN] {split}: filtered {original_len - len(self.images)} missing images")

        if len(self.images) == 0:
            raise ValueError(f"Split '{split}' is empty after filtering!")

        # Augmentation: only applied to the training split.
        if augment and split == 'train':
            self.transform = build_train_transform(image_size=image_size)
        else:
            self.transform = None

        self._seed = seed + 1
        self._rng = random.Random(seed + 1)
        print(f"{split} split: {len(self.images)} images "
              f"(augment={'on' if self.transform is not None else 'off'})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = resolve_image_path(self.image_dir, img_name)
        if img_path is None:
            raise FileNotFoundError(f"Missing image file: {img_name}")

        with Image.open(img_path) as img:
            image = img.convert('RGB')

        # Apply augmentation (train only) before the processor.
        if self.transform is not None:
            image = self.transform(image)

        captions = self.image_captions[img_name]
        caption = self._rng.choice(captions) if self.split == 'train' else captions[0]

        image_input = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)

        return {
            'image': image_input,
            'caption': caption,
            'image_name': img_name
        }


# Collation
def collate_fn(batch, processor):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_names = [item['image_name'] for item in batch]

    text_outputs = processor(text=captions, return_tensors='pt', padding=True, truncation=True, max_length=77, return_attention_mask=True)

    return {
        'image': images,
        'text': text_outputs['input_ids'],
        'attention_mask': text_outputs['attention_mask'],
        'caption': captions,
        'image_name': image_names
    }


# DataLoaders
def create_dataloaders(image_dir, captions_file, processor, batch_size=32, seed=42, num_workers=4, min_caption_len=MIN_CAPTION_LEN,
                       augment=True, image_size=224):

    train_dataset = Flickr8kDataset(image_dir, captions_file, processor, 'train', seed=seed, min_caption_len=min_caption_len,
        augment=augment, image_size=image_size)
    # Validation and test: no augmentation (deterministic evaluation).
    val_dataset = Flickr8kDataset(image_dir, captions_file, processor, 'val', seed=seed, min_caption_len=min_caption_len, augment=False)
    test_dataset = Flickr8kDataset(image_dir, captions_file, processor, 'test', seed=seed, min_caption_len=min_caption_len, augment=False)

    g = torch.Generator()
    g.manual_seed(seed)

    collate = partial(collate_fn, processor=processor)

    common = dict(
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader