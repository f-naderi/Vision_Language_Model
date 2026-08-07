import torch
import csv
import random
from PIL import Image
import matplotlib.pyplot as plt

def run_inference(model, test_loader, IMAGE_DIR, CAPTIONS_FILE, device='cuda'):
    """
    Run inference demo on a random test image.
    
    Args:
        model: Fine-tuned CLIP model
        test_loader: Test data loader
        IMAGE_DIR: Path to image folder
        CAPTIONS_FILE: Path to captions file
        device: 'cuda' or 'cpu'
    """
    
    # Load best model
    print("Loading best model...")
    checkpoint = torch.load('outputs/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f" Loaded from epoch {checkpoint['epoch']+1}, Val Loss: {checkpoint['val_loss']:.4f}")

    # Get a random test image
    test_batch = next(iter(test_loader))
    idx = random.randint(0, len(test_batch['image']) - 1)
    sample_image_name = test_batch['image_name'][idx]
    sample_image_path = f"{IMAGE_DIR}/{sample_image_name}"

    # Get ground truth captions
    ground_truth_captions = []
    with open(CAPTIONS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0].strip() == sample_image_name:
                ground_truth_captions.append(row[1].strip())

    print(f"\n{'-'*60}")
    print(f"Test Image: {sample_image_name}")
    print(f"{'-'*60}")
    print(f"Ground Truth Captions:")
    for gt in ground_truth_captions:
        print(f"  • {gt}")

    # Display image
    img = Image.open(sample_image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Test Image', fontsize=14, fontweight='bold')
    plt.show()

    # ==========================================
    # Inference with 100 random captions
    # ==========================================
    
    # Load all captions
    all_captions = []
    with open(CAPTIONS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        all_captions = [(row[0].strip(), row[1].strip()) for row in reader]

    # Get random captions + ground truth
    random.seed(42)
    sample_captions = random.sample(all_captions, min(100, len(all_captions)))
    for gt in ground_truth_captions:
        if (sample_image_name, gt) not in sample_captions:
            sample_captions.append((sample_image_name, gt))

    caption_texts = [cap[1] for cap in sample_captions]
    print(f"\nSearching among {len(caption_texts)} captions...")

    # Get processor for text
    processor = model.get_preprocess()

    model.eval()
    with torch.no_grad():
        image_tensor = test_batch['image'][idx:idx+1].to(device)
        
        # Process all captions
        text_inputs = processor(
            text=caption_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77
        )['input_ids'].to(device)
        
        # Get features
        _, _, img_feat, txt_feat = model(image_tensor, text_inputs)
        similarities = (100.0 * img_feat @ txt_feat.T).squeeze()
        probabilities = torch.softmax(similarities, dim=0)
        
        # Top-5 results
        top5 = similarities.topk(min(5, len(caption_texts)))
        
        print(f"\n{'-'*60}")
        print(f" Top-5 Retrieved Captions:")
        for rank, (idx_pred, score) in enumerate(zip(top5.indices, top5.values)):
            caption = caption_texts[idx_pred]
            prob = probabilities[idx_pred].item()
            is_gt = " GROUND TRUTH" if caption in ground_truth_captions else ""
            print(f"  {rank+1}. [{score:.1f}] ({prob:.1%}) {caption[:100]} {is_gt}")
        
        # Check Recall
        top5_captions = [caption_texts[i] for i in top5.indices]
        recall_at_5 = sum(1 for gt in ground_truth_captions if gt in top5_captions)
        
        print(f"\n{'-'*60}")
        print(f" Recall@5: {recall_at_5}/5 ground truth captions found")
        print(f"   Best score: {similarities[top5.indices[0]].item():.1f}")
    
    return top5, caption_texts, ground_truth_captions