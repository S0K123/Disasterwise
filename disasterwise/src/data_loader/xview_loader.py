import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from shapely.wkt import loads as wkt_loads
from typing import Dict, List, Any, Optional

class XViewDataset(Dataset):
    """
    Dataset class for xView2 (xBD) satellite imagery.
    Supports parsing of building polygons and damage classification.
    """
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        transform: Optional[Any] = None,
        target_format: str = 'bbox' # 'bbox' or 'mask'
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_format = target_format
        
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        
        # Mapping for damage levels
        self.damage_map = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3,
            'un-classified': -1
        }

    def __len__(self) -> int:
        return len(self.image_files)

    def _parse_annotation(self, label_path: str) -> Dict[str, Any]:
        """
        Parses xBD JSON annotation file.
        """
        with open(label_path, 'r') as f:
            data = json.load(f)
            
        features = data.get('features', {}).get('xy', [])
        metadata = data.get('metadata', {})
        
        objs = []
        for feat in features:
            properties = feat.get('properties', {})
            wkt = feat.get('wkt', '')
            
            if not wkt:
                continue
                
            # Parse WKT to polygon
            poly = wkt_loads(wkt)
            
            # Convert polygon to bounding box (minx, miny, maxx, maxy)
            bbox = list(poly.bounds)
            
            # Get class/subtype
            feature_type = properties.get('feature_type', 'building')
            subtype = properties.get('subtype', 'no-damage')
            damage_level = self.damage_map.get(subtype, -1)
            
            objs.append({
                'bbox': bbox,
                'category': feature_type,
                'damage_level': damage_level,
                'poly': poly
            })
            
        return {
            'objects': objs,
            'metadata': metadata
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_name = img_name.replace('.png', '.json')
        label_path = os.path.join(self.labels_dir, label_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load and parse annotation
        target = {}
        if os.path.exists(label_path):
            anno = self._parse_annotation(label_path)
            target['objects'] = anno['objects']
            target['metadata'] = anno['metadata']
        else:
            target['objects'] = []
            target['metadata'] = {}

        if self.transform:
            # Albumentations or other transforms
            # Note: Need to handle bbox/mask transformation if used in training
            image_np = np.array(image)
            transformed = self.transform(image=image_np)
            image = transformed['image']
            
        return {
            'image': image,
            'target': target,
            'image_id': img_name
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculates basic statistics about the dataset split.
        """
        total_images = len(self.image_files)
        total_buildings = 0
        damage_counts = {k: 0 for k in self.damage_map.keys()}
        
        for img_name in self.image_files:
            label_name = img_name.replace('.png', '.json')
            label_path = os.path.join(self.labels_dir, label_name)
            if os.path.exists(label_path):
                anno = self._parse_annotation(label_path)
                objs = anno['objects']
                total_buildings += len(objs)
                for obj in objs:
                    level = [k for k, v in self.damage_map.items() if v == obj['damage_level']]
                    if level:
                        damage_counts[level[0]] += 1
                        
        return {
            'split': self.split,
            'total_images': total_images,
            'total_buildings': total_buildings,
            'damage_distribution': damage_counts
        }
