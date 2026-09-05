import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(config, mode='train'):
    """
    Returns data transformations based on the config.
    """
    image_size = config['preprocessing']['image_size']
    
    if mode == 'train':
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

class Preprocessor:
    """
    Handles data preprocessing steps.
    """
    def __init__(self, config):
        self.config = config

    def preprocess_image(self, image_path):
        """
        Loads and applies basic preprocessing to a single image.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # TODO: Add more custom preprocessing if needed
        return image
