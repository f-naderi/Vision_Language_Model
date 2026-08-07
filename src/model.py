import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPForFlickr8k(nn.Module):
    def __init__(self, clip_model_name='models/clip-vit-base-patch32', device='cuda'):
        super().__init__()
        self.device = device
        
        # Load from local folder
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
        
        # Freeze all parameters first
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last layer of vision encoder
        for param in self.clip_model.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        
        # Unfreeze last layer of text encoder
        for param in self.clip_model.text_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.clip_model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def forward(self, images, texts):
        """Forward pass"""
        with torch.amp.autocast('cuda'):
            # Get image features
            image_outputs = self.clip_model.get_image_features(pixel_values=images)
            if hasattr(image_outputs, 'pooler_output'):
                image_features = image_outputs.pooler_output
            else:
                image_features = image_outputs
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get text features
            text_outputs = self.clip_model.get_text_features(input_ids=texts)
            if hasattr(text_outputs, 'pooler_output'):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Use learned logit_scale
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
        return logits_per_image, logits_per_text, image_features, text_features
    
    def get_preprocess(self):
        """Return processor for preprocessing"""
        return self.processor