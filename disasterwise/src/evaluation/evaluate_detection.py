import os
import yaml
import argparse
import torch
from ultralytics import RTDETR
from src.utils.logger import setup_logger
from src.data_loader.xview_loader import XViewDataset
from torch.utils.data import DataLoader
from typing import Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR for Disaster Detection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained RT-DETR model (.pt)")
    parser.add_argument("--data_split", type=str, default="test", help="Data split to evaluate on (val or test)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    logger = setup_logger(
        log_file=config['logging']['log_file'],
        level=config['logging']['level']
    )
    
    logger.info(f"Evaluating detection model on the {args.data_split} split...")
    
    # Load model
    model = RTDETR(args.model)
    
    # Ultralytics' validation function handles mAP, precision, and recall
    # It requires the data to be configured in a YAML file
    data_yaml_path = os.path.join(os.path.dirname(args.config), 'rtdetr_data.yaml')
    
    if not os.path.exists(data_yaml_path):
        logger.error(f"Data config {data_yaml_path} not found. Please run training first to generate it.")
        return

    metrics = model.val(
        data=data_yaml_path,
        split=args.data_split,
        project="experiments/metrics",
        name="detection_evaluation"
    )
    
    logger.info("Detection Evaluation Complete.")
    logger.info(f"Results saved to experiments/metrics/detection_evaluation")
    logger.info(f"mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"mAP50: {metrics.box.map50:.4f}")
    logger.info(f"mAP75: {metrics.box.map75:.4f}")

if __name__ == "__main__":
    main()
