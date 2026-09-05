import os
import yaml
import json
from src.data_loader.xview_loader import XViewDataset

def test_loader():
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    raw_dir = config['data']['xview_dir']
    print(f"Testing loader on {raw_dir}...")
    
    try:
        dataset = XViewDataset(root_dir=raw_dir, split='train')
        print(f"Total images in train: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Successfully loaded image: {sample['image_id']}")
            print(f"Metadata: {sample['target']['metadata']}")
            print(f"Number of objects: {len(sample['target']['objects'])}")
            if len(sample['target']['objects']) > 0:
                print(f"Sample object: {sample['target']['objects'][0]}")
        
        # Get stats
        print("Calculating dataset stats (this might take a while for large datasets)...")
        # For testing, let's just do it for the first 10 images if it's too large
        # But XViewDataset.get_stats() does it for all.
        # Let's override it or just run it if the dataset is small.
        # Based on previous LS, there are many files.
        
        stats = dataset.get_stats()
        print(f"Stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        print(f"Error testing loader: {e}")

if __name__ == "__main__":
    print("Starting test...")
    test_loader()
