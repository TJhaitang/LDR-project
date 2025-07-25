import os

import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from typing import List

warnings.filterwarnings('ignore')

class VisionProj(nn.Module):
    def __init__(self, ve_hidden_size=768, hidden_size=512):
        super().__init__()
        self.ve_hidden_size = ve_hidden_size
        self.hidden_size = hidden_size
        self.vision_proj = nn.Sequential(
            nn.Linear(self.ve_hidden_size, self.hidden_size)
        )

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj
    
class VisionEncoder(nn.Module):
    def __init__(self, vision_model_path="./model/vision_model/clip-vit-base-patch16"):
        super().__init__()
        self.vision_encoder = self.get_vision_model(vision_model_path)
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        #hidden_size需要进一步适配语言模型
        self.vision_proj = VisionProj(ve_hidden_size=768, hidden_size=512)

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = CLIPModel.from_pretrained(model_path)
        # processor = CLIPProcessor.from_pretrained(model_path)
        return model
    
    def forward(self, images: torch.Tensor):
        if self.vision_encoder is None:
            raise ValueError("Vision encoder model not found.")
        image_features = self.vision_encoder.get_image_features(images)
        image_encoders = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.vision_proj(image_encoders)