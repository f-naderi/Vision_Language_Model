import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch in progress_bar:
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits_per_image, logits_per_text, _, _ = model(images, texts)
            
            # Symmetric CLIP loss
            ground_truth = torch.arange(len(images), device=device)
            loss_imgs = criterion(logits_per_image, ground_truth)
            loss_txts = criterion(logits_per_text, ground_truth)
            loss = (loss_imgs + loss_txts) / 2
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits_per_image, logits_per_text, _, _ = model(images, texts)
                
                ground_truth = torch.arange(len(images), device=device)
                loss_imgs = criterion(logits_per_image, ground_truth)
                loss_txts = criterion(logits_per_text, ground_truth)
                loss = (loss_imgs + loss_txts) / 2
                
                total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train(model, train_loader, val_loader, num_epochs=10, lr=1e-5, device='cuda'):
    # FIX: Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizing {sum(p.numel() for p in trainable_params):,} parameters")
    
    # FIX: Add weight decay
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    # FIX: Add cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.1)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Create output directory
    os.makedirs('outputs/checkpoints', exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, 'outputs/checkpoints/best_model.pth')
            print(" Best model saved!")
    
    return train_losses, val_losses