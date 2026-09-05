import os
import json
import yaml
import argparse
import torch
from ultralytics import RTDETR
from src.utils.logger import setup_logger
from src.utils.experiment import ExperimentTracker
from typing import Dict, Any, List

def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR for Disaster Detection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="Pre-trained RT-DETR weights")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for training")
    parser.add_argument("--batch", type=int, help="Batch size (overrides config)")
    return parser.parse_args()

def convert_xbd_to_yolo(json_path: str, output_path: str, img_size: int = 1024):
    """
    Converts xBD JSON annotation to YOLO format for RT-DETR.
    Classes: 0: building, 1: damaged_structure, 2: infrastructure
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    yolo_lines = []
    # Map xBD damage levels to our 3 target classes
    # 0: building, 1: damaged_structure, 2: infrastructure
    
    for feat in data.get('features', {}).get('xy', []):
        properties = feat.get('properties', {})
        wkt = feat.get('wkt', '')
        if not wkt: continue
        
        # Simple extraction of bbox from WKT for YOLO conversion
        # RT-DETR expects [class, x_center, y_center, width, height] normalized
        from shapely.wkt import loads as wkt_loads
        poly = wkt_loads(wkt)
        minx, miny, maxx, maxy = poly.bounds
        
        x_center = (minx + maxx) / 2 / img_size
        y_center = (miny + maxy) / 2 / img_size
        width = (maxx - minx) / img_size
        height = (maxy - miny) / img_size
        
        feature_type = properties.get('feature_type', 'building')
        subtype = properties.get('subtype', 'no-damage')
        
        class_id = 0 # Default: building
        if subtype in ['major-damage', 'destroyed']:
            class_id = 1 # damaged_structure
        elif feature_type == 'infrastructure':
            class_id = 2 # infrastructure
            
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

def generate_ultralytics_yaml(config: Dict[str, Any], output_path: str):
    """
    Generates a YAML file for Ultralytics training from project config.
    """
    processed_dir = config['data']['processed_dir']
    
    data_yaml = {
        'path': os.path.abspath(processed_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'building',
            1: 'damaged_structure',
            2: 'infrastructure'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return output_path

def prepare_yolo_labels(config: Dict[str, Any], logger: Any):
    """
    Prepares labels in YOLO format for each split.
    """
    processed_dir = config['data']['processed_dir']
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(processed_dir, split)
        labels_src = os.path.join(split_dir, 'labels')
        labels_dest = os.path.join(split_dir, 'labels_yolo')
        
        if not os.path.exists(labels_src):
            continue
            
        os.makedirs(labels_dest, exist_ok=True)
        logger.info(f"Converting annotations to YOLO format for {split} split...")
        
        for lbl_file in os.listdir(labels_src):
            if lbl_file.endswith('.json'):
                src_path = os.path.join(labels_src, lbl_file)
                dest_path = os.path.join(labels_dest, lbl_file.replace('.json', '.txt'))
                convert_xbd_to_yolo(src_path, dest_path)

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Experiment Tracking
    tracker = ExperimentTracker(config, experiment_name="rtdetr_disaster_detection")
    
    # Setup logger
    logger = setup_logger(
        log_file=tracker.get_log_file_path(),
        level=config['logging']['level']
    )
    
    logger.info("Initializing RT-DETR Training Pipeline...")
    
    # Prepare YOLO labels (required for Ultralytics/RT-DETR)
    prepare_yolo_labels(config, logger)
    
    # Overrides from CLI
    epochs = args.epochs or config['model']['parameters'].get('epochs', 50)
    batch = args.batch or config['model']['parameters'].get('batch_size', 8)
    
    # Generate data config for Ultralytics
    data_yaml_path = os.path.join(os.path.dirname(args.config), 'rtdetr_data.yaml')
    generate_ultralytics_yaml(config, data_yaml_path)
    
    # Initialize and train RT-DETR model
    logger.info(f"Loading RT-DETR model: {args.model}")
    model = RTDETR(args.model)
    
    logger.info(f"Starting training for {epochs} epochs with batch size {batch}...")
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        project=os.path.dirname(tracker.exp_dir),
        name=os.path.basename(tracker.exp_dir),
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info("Training complete. Evaluating on test set...")
    metrics = model.val(data=data_yaml_path, split='test')
    
    # Log metrics
    tracker.log_metrics({
        'mAP50-95': metrics.box.map,
        'mAP50': metrics.box.map50,
        'mAP75': metrics.box.map75
    })
    
    logger.info(f"Test mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
