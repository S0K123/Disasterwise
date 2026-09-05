import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Tuple
from src.utils.logger import setup_logger

class FeatureExtractor:
    """
    Research-grade feature extraction pipeline for satellite imagery.
    Converts object detection outputs and spatial metadata into numerical 
    feature vectors suitable for Transformer-based risk prediction.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(log_file='experiments/feature_extraction.log', level='INFO')
        self.imgsz = config['model']['parameters'].get('imgsz', 1024)
        self.total_area = self.imgsz * self.imgsz

    def compute_spatial_stats(self, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes spatial statistics from detection results.
        Extracted features: building_density, vehicle_density, infrastructure_count, open_area_ratio.
        """
        stats = {
            'building_density': 0.0,
            'vehicle_density': 0.0,
            'infrastructure_count': 0,
            'open_area_ratio': 1.0,
            'damaged_ratio': 0.0,
            'flood_ratio': 0.0
        }
        
        building_area = 0.0
        occupied_area = 0.0
        damaged_buildings = 0
        total_buildings = 0
        
        for det in detections:
            label = det['label'].lower()
            bbox = det['bbox'] # [x1, y1, x2, y2]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            occupied_area += area
            
            if 'building' in label:
                total_buildings += 1
                building_area += area
                if 'damaged' in label:
                    damaged_buildings += 1
            
            if 'vehicle' in label:
                stats['vehicle_density'] += 1
                
            if 'infrastructure' in label:
                stats['infrastructure_count'] += 1
                
            if 'flood' in label:
                stats['flood_ratio'] += (area / self.total_area)

        # Normalize counts to densities
        stats['building_density'] = building_area / self.total_area
        stats['vehicle_density'] = stats['vehicle_density'] / self.total_area
        stats['open_area_ratio'] = max(0.0, 1.0 - (occupied_area / self.total_area))
        stats['damaged_ratio'] = damaged_buildings / total_buildings if total_buildings > 0 else 0.0
        
        return stats

    def detections_to_vector(self, detections: List[Dict[str, Any]], environmental_data: Dict[str, float], temporal_features: Dict[str, float] = None) -> np.ndarray:
        """
        Converts detection outputs and environmental data into a fixed-length feature vector.
        Matches the input_dim required by the Transformer model.
        """
        spatial_stats = self.compute_spatial_stats(detections)
        
        # Combine all features into a single vector
        # Order must be consistent for the model
        feature_vector = [
            spatial_stats['building_density'],
            spatial_stats['vehicle_density'],
            spatial_stats['infrastructure_count'],
            spatial_stats['open_area_ratio'],
            spatial_stats['damaged_ratio'],
            spatial_stats['flood_ratio'],
            environmental_data.get('temp', 0.0),
            environmental_data.get('humidity', 0.0),
            environmental_data.get('wind_speed', 0.0),
            environmental_data.get('rainfall', 0.0),
            environmental_data.get('pressure', 0.0),
            environmental_data.get('cloud_cover', 0.0),
            environmental_data.get('vegetation_index', 0.0),
            environmental_data.get('soil_moisture', 0.0),
            environmental_data.get('elevation', 0.0),
            environmental_data.get('pop_density', 0.0),
            environmental_data.get('dist_to_coast', 0.0),
            environmental_data.get('slope', 0.0),
            environmental_data.get('visibility', 0.0),
            environmental_data.get('historical_risk', 0.0)
        ]
        
        if temporal_features:
            feature_vector.extend([
                temporal_features.get('building_damage_ratio', 0.0),
                temporal_features.get('infrastructure_change', 0.0),
                temporal_features.get('debris_increase', 0.0),
                temporal_features.get('vegetation_loss', 0.0)
            ])
        
        return np.array(feature_vector, dtype=np.float32)

    def process_batch(self, batch_detections: List[List[Dict[str, Any]]], batch_env: List[Dict[str, float]]) -> torch.Tensor:
        """
        Processes a batch of detections and environmental data into a tensor for the Transformer.
        """
        vectors = []
        for detections, env in zip(batch_detections, batch_env):
            vectors.append(self.detections_to_vector(detections, env))
            
        return torch.tensor(np.array(vectors))

if __name__ == "__main__":
    # Example usage
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    extractor = FeatureExtractor(config)
    
    # Mock detection output
    mock_detections = [
        {'label': 'building', 'bbox': [100, 100, 200, 200], 'confidence': 0.9},
        {'label': 'damaged_building', 'bbox': [300, 300, 400, 400], 'confidence': 0.8},
        {'label': 'flooded_area', 'bbox': [0, 0, 500, 500], 'confidence': 0.75}
    ]
    
    # Mock environmental data
    mock_env = {'temp': 25.5, 'humidity': 0.8, 'wind_speed': 12.0, 'rainfall': 5.0}
    
    vector = extractor.detections_to_vector(mock_detections, mock_env)
    print(f"Extracted Feature Vector (dim={len(vector)}):")
    print(vector)
