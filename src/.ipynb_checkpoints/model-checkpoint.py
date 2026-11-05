import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CLIPForFlickr8k(nn.Module):
    def __init__(self, clip_model_name='models/clip-vit-base-patch32', device='cuda'):
        super().__init__()

        print(f"Loading CLIP from: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)

        # ---------------------------------------------------------------
        # Freeze everything first
        # ---------------------------------------------------------------
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze last transformer layer of each encoder
        for param in self.clip_model.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in self.clip_model.text_model.encoder.layers[-1].parameters():
            param.requires_grad = True

        # Unfreeze projection layers
        for param in self.clip_model.visual_projection.parameters():
            param.requires_grad = True
        for param in self.clip_model.text_projection.parameters():
            param.requires_grad = True

        # Freeze logit_scale
        with torch.no_grad():
            self.clip_model.logit_scale.fill_(2.6592)
        self.clip_model.logit_scale.requires_grad = False
        # print(f"logit_scale raw: {self.clip_model.logit_scale.item():.4f}")
        # print(f"logit_scale exp: {self.clip_model.logit_scale.exp().item():.4f}")

        trainable = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.clip_model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

    # ---------------------------------------------------------------
    # Feature extractors
    # ---------------------------------------------------------------
    def _extract_image_features(self, pixel_values):
        
        if hasattr(self.clip_model, 'get_image_features'):
            feats = self.clip_model.get_image_features(pixel_values=pixel_values)
            if isinstance(feats, torch.Tensor):
                return F.normalize(feats, p=2, dim=-1)
        
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        feats = self.clip_model.visual_projection(pooled)
        return F.normalize(feats, p=2, dim=-1)

    def _extract_text_features(self, input_ids, attention_mask=None):
        
        if hasattr(self.clip_model, 'get_text_features'):
            feats = self.clip_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(feats, torch.Tensor):
                return F.normalize(feats, p=2, dim=-1)
        
        text_outputs = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = text_outputs.pooler_output
        feats = self.clip_model.text_projection(pooled)
        return F.normalize(feats, p=2, dim=-1)

    def get_logit_scale(self):
        """Return logit_scale with the same safe clamp as original CLIP."""
        return self.clip_model.logit_scale.exp().clamp(max=100.0)


    @torch.no_grad()
    def encode_images(self, images):
        return self._extract_image_features(images)

    @torch.no_grad()
    def encode_texts(self, input_ids, attention_mask=None):
        return self._extract_text_features(input_ids, attention_mask)

    def forward(self, images, texts, attention_mask=None):
        """Return CLIP logits + normalized features."""
        image_features = self._extract_image_features(images)
        text_features = self._extract_text_features(texts, attention_mask)

        logit_scale = self.get_logit_scale()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, image_features, text_features

    def get_preprocess(self):
        return self.processor