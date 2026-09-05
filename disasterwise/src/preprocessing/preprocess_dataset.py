import os
import json
import shutil
import argparse
import yaml
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from src.data_loader.xview_loader import XViewDataset
from src.utils.logger import setup_logger

def setup_args():
    parser = argparse.ArgumentParser(description="xView2 Dataset Preprocessing and Splitting")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--raw_dir", type=str, help="Path to raw data directory")
    parser.add_argument("--output_dir", type=str, help="Path to output processed directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--tile_size", type=int, default=None, help="Size of tiles if image tiling is needed")
    return parser.parse_args()

def tile_image(image_np: np.ndarray, tile_size: int) -> List[np.ndarray]:
    """
    Splits a large image into smaller tiles.
    """
    h, w, c = image_np.shape
    tiles = []
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = image_np[i:i+tile_size, j:j+tile_size, :]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tiles.append(tile)
    return tiles

def validate_data_pairs(image_files: List[str]) -> List[Tuple[str, str]]:
    """
    Validates pre and post disaster image pairs.
    Each pre image should have a corresponding post image.
    """
    pre_images = sorted([f for f in image_files if '_pre_disaster' in f])
    post_images = sorted([f for f in image_files if '_post_disaster' in f])
    
    pairs = []
    for pre in pre_images:
        post = pre.replace('_pre_disaster', '_post_disaster')
        if post in post_images:
            pairs.append((pre, post))
            
    return pairs

def process_and_save_split(
    pairs: List[Tuple[str, str]], 
    src_images_dir: str, 
    src_labels_dir: str, 
    dest_dir: str,
    split_name: str,
    logger: logging.Logger
):
    """
    Copies images and labels to the split directory.
    """
    split_path = os.path.join(dest_dir, split_name)
    img_dest = os.path.join(split_path, 'images')
    lbl_dest = os.path.join(split_path, 'labels')
    
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(lbl_dest, exist_ok=True)
    
    logger.info(f"Processing {split_name} split with {len(pairs)} pairs...")
    
    for pre, post in pairs:
        # Copy images
        shutil.copy2(os.path.join(src_images_dir, pre), os.path.join(img_dest, pre))
        shutil.copy2(os.path.join(src_images_dir, post), os.path.join(img_dest, post))
        
        # Copy labels
        pre_lbl = pre.replace('.png', '.json')
        post_lbl = post.replace('.png', '.json')
        
        if os.path.exists(os.path.join(src_labels_dir, pre_lbl)):
            shutil.copy2(os.path.join(src_labels_dir, pre_lbl), os.path.join(lbl_dest, pre_lbl))
        if os.path.exists(os.path.join(src_labels_dir, post_lbl)):
            shutil.copy2(os.path.join(src_labels_dir, post_lbl), os.path.join(lbl_dest, post_lbl))
            
    logger.info(f"Successfully saved {split_name} split to {split_path}")

def main():
    args = setup_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override from args if provided
    raw_dir = args.raw_dir or config['data']['xview_dir']
    output_dir = args.output_dir or config['data']['processed_dir']
    
    # Setup logging
    logger = setup_logger(
        log_file=os.path.join(config['logging']['log_file']),
        level=config['logging']['level']
    )
    
    logger.info("Starting xView2 dataset preprocessing...")
    
    # Use the train folder from raw data
    src_images_dir = os.path.join(raw_dir, 'train', 'images')
    src_labels_dir = os.path.join(raw_dir, 'train', 'labels')
    
    if not os.path.exists(src_images_dir):
        logger.error(f"Source images directory not found: {src_images_dir}")
        return
        
    image_files = os.listdir(src_images_dir)
    pairs = validate_data_pairs(image_files)
    
    logger.info(f"Found {len(pairs)} valid pre-post image pairs.")
    
    # Perform splitting
    train_pairs, test_pairs = train_test_split(pairs, test_size=args.test_size, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=args.val_size / (1 - args.test_size), random_state=42)
    
    logger.info(f"Splits: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Process and save each split
    process_and_save_split(train_pairs, src_images_dir, src_labels_dir, output_dir, 'train', logger)
    process_and_save_split(val_pairs, src_images_dir, src_labels_dir, output_dir, 'val', logger)
    process_and_save_split(test_pairs, src_images_dir, src_labels_dir, output_dir, 'test', logger)
    
    # Calculate and log statistics for the processed train set
    processed_dataset = XViewDataset(root_dir=output_dir, split='train')
    stats = processed_dataset.get_stats()
    logger.info(f"Dataset Statistics (Train): {json.dumps(stats, indent=2)}")
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
