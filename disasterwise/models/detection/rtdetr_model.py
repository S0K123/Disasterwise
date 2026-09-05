import torch
import torch.nn as nn
from ultralytics import RTDETR
from typing import List, Dict, Any, Optional

class DisasterRTDETR(nn.Module):
    """
    Research-grade wrapper for Real-Time DEtection TRansformer (RT-DETR)
    specifically tuned for disaster-related object detection in satellite imagery.
    """
    def __init__(
        self, 
        model_path: str = 'rtdetr-l.pt', 
        num_classes: int = 5,
        imgsz: int = 1024
    ):
        super(DisasterRTDETR, self).__init__()
        # Load pre-trained RT-DETR from Ultralytics
        self.model = RTDETR(model_path)
        self.imgsz = imgsz
        
        # Disaster class labels (specifically for buildings and infrastructure)
        self.classes = {
            0: 'building',
            1: 'damaged_structure',
            2: 'infrastructure'
        }

    def forward(self, x: torch.Tensor) -> Any:
        """
        Forward pass for the model.
        Note: Ultralytics models handle their own forward pass internally
        when called via the .predict() or .train() methods.
        """
        return self.model(x)

    def load_weights(self, path: str):
        """
        Loads custom trained weights.
        """
        self.model = RTDETR(path)

    def save_model(self, path: str):
        """
        Saves the model weights.
        """
        self.model.save(path)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the model configuration.
        """
        return {
            'model_type': 'RT-DETR',
            'imgsz': self.imgsz,
            'classes': self.classes,
            'num_classes': len(self.classes)
        }
