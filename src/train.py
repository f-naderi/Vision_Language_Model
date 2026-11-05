import torch
import torch.nn as nn
from tqdm import tqdm
import os


def _autocast_ctx(device):
    """Returns an autocast context manager for mixed-precision training."""
    use_cuda = torch.cuda.is_available() and device.startswith('cuda')
    return torch.amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=use_cuda)


def train_epoch(model, train_loader, optimizer, criterion, device, scaler, max_grad_norm=1.0):
    """Runs one full training epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc='Training')

    # Only optimize parameters that require gradients (supports frozen backbones).
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch in progress_bar:
        # Move batch to device. non_blocking=True is safe with pinned memory.
        images = batch['image'].to(device, non_blocking=True)
        texts = batch['text'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)

        # Zero gradients. set_to_none=True is faster and saves memory.
        optimizer.zero_grad(set_to_none=True)

        # Single autocast context
        with _autocast_ctx(device):
            # Forward pass returns similarity logits in both directions.
            logits_per_image, logits_per_text, _, _ = model(images, texts, attention_mask)

            # Contrastive loss (InfoNCE): ground truth is the diagonal (i-th image ↔ i-th text).
            ground_truth = torch.arange(len(images), device=device)
            loss_imgs = criterion(logits_per_image, ground_truth)  # image → text
            loss_txts = criterion(logits_per_text, ground_truth)  # text → image
            # Symmetric loss: average of both directions.
            loss = (loss_imgs + loss_txts) / 2

        # Backward pass with gradient scaling (for AMP stability).
        scaler.scale(loss).backward()

        # ---- Gradient clipping ----
        # Must call unscale_ before clipping, otherwise gradients are still scaled.
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

        # Optimizer step + scaler update.
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss weighted by batch size (to compute exact epoch average).
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total_samples


def validate(model, val_loader, criterion, device):
    """Evaluates the model on the validation set (no gradient updates)."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for efficiency.
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(device, non_blocking=True)
            texts = batch['text'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            with _autocast_ctx(device):
                logits_per_image, logits_per_text, _, _ = model(images, texts, attention_mask)

                # Same symmetric contrastive loss as in training.
                ground_truth = torch.arange(len(images), device=device)
                loss_imgs = criterion(logits_per_image, ground_truth)
                loss_txts = criterion(logits_per_text, ground_truth)
                loss = (loss_imgs + loss_txts) / 2

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples


def train(model, train_loader, val_loader, num_epochs=10, lr=1e-5, device='cuda', max_grad_norm=1.0):

    # Collect trainable parameters (supports partial fine-tuning).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizing {sum(p.numel() for p in trainable_params):,} parameters")

    # AdamW is the standard optimizer for CLIP-style contrastive training.
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    # Cosine annealing for smooth LR decay.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.1)
    # CrossEntropyLoss is used for InfoNCE (each row is a softmax over candidates).
    criterion = nn.CrossEntropyLoss()

    # GradScaler for AMP. Enabled only on CUDA.
    use_cuda = torch.cuda.is_available() and device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    # Ensure checkpoint directory exists.
    os.makedirs('outputs/checkpoints', exist_ok=True)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # ---- Training ----
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, max_grad_norm=max_grad_norm)
        # ---- Validation ----
        val_loss = validate(model, val_loader, criterion, device)

        # Update LR after each epoch.
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")

        # Save checkpoint if validation loss improved.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, 'outputs/checkpoints/best_model.pth')
            print("Best model saved")

    # ---- Load the best checkpoint at the end ----
    print("\n" + "-" * 60)
    print("Loading best model weights...")
    checkpoint = torch.load('outputs/checkpoints/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded from epoch {checkpoint['epoch'] + 1}, Val Loss: {checkpoint['val_loss']:.4f}")
    print("-" * 60)

    return train_losses, val_losses