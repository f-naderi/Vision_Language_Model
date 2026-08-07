import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import csv
from collections import defaultdict

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, processor, split='train'):
        """
        Args:
            image_dir: Path to images folder
            captions_file: Path to captions file
            processor: CLIP processor
            split: 'train', 'val', or 'test'
        """
        self.image_dir = image_dir
        self.processor = processor
        
        # Read captions - supports both CSV and TSV formats
        self.image_captions = defaultdict(list)
        self.images = []
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            # Detect delimiter (comma or tab)
            first_line = f.readline().strip()
            f.seek(0)
            
            if '\t' in first_line:
                delimiter = '\t'
            else:
                delimiter = ','
            
            # Try reading as CSV
            reader = csv.reader(f, delimiter=delimiter)
            
            # Skip header if exists
            first_row = next(reader)
            if first_row[0].lower() in ['image', 'image_name', 'img']:
                pass  # Skip header
            else:
                # Process first row as data
                if len(first_row) >= 2:
                    img_name = first_row[0].strip()
                    caption = first_row[1].strip()
                    self.image_captions[img_name].append(caption)
            
            # Process remaining rows
            for row in reader:
                if len(row) >= 2:
                    img_name = row[0].strip()
                    caption = row[1].strip()
                    self.image_captions[img_name].append(caption)
        
        
        # Split data
        unique_images = list(self.image_captions.keys())
        total = len(unique_images)
        
        if total == 0:
            raise ValueError("No images found in captions file!")
        
        if split == 'train':
            self.images = unique_images[:int(0.8 * total)]
        elif split == 'val':
            self.images = unique_images[int(0.8 * total):int(0.9 * total)]
        else:  # test
            self.images = unique_images[int(0.9 * total):]
        
        print(f"{split} split: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Check if file exists, if not try adding extensions
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_path = img_path + ext
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')
        
        # Randomly select one caption
        import random
        captions = self.image_captions[img_name]
        caption = random.choice(captions)
        
        # Process image (always same size: 224x224)
        image_input = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        
        # Process text with padding to max_length=77
        text_input = self.processor(text=caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77)['input_ids'].squeeze(0)
        
        return {'image': image_input, 'text': text_input, 'caption': caption, 'image_name': img_name}

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    images = torch.stack([item['image'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    
    return {'image': images, 'text': texts, 'caption': captions, 'image_name': image_names}

def create_dataloaders(image_dir, captions_file, processor, batch_size=32):
    """Create train, val, test dataloaders"""
    train_dataset = Flickr8kDataset(image_dir, captions_file, processor, 'train')
    val_dataset = Flickr8kDataset(image_dir, captions_file, processor, 'val')
    test_dataset = Flickr8kDataset(image_dir, captions_file, processor, 'test')
    
    # Use custom collate_fn and num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader