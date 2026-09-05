import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from ultralytics import RTDETR
from typing import Dict, Any, List
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with RT-DETR for Disaster Detection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained RT-DETR model (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to satellite image (.png, .jpg, .tif)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save the result visualization")
    return parser.parse_args()

def run_inference(
    model: RTDETR, 
    image_path: str, 
    imgsz: int = 1024, 
    conf: float = 0.25, 
    save: bool = False
) -> Dict[str, Any]:
    """
    Runs RT-DETR inference on a single satellite image.
    """
    # Run prediction
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        save=save,
        project='experiments/inference',
        name='disaster_detection',
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    
    # Process results
    result = results[0]
    boxes = result.boxes
    
    # Extract labels and coordinates
    detections = []
    for box in boxes:
        coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        cls_id = int(box.cls[0].item())
        conf_score = box.conf[0].item()
        
        detections.append({
            'label': result.names[cls_id],
            'bbox': coords,
            'confidence': conf_score
        })
        
    return {
        'image_id': os.path.basename(image_path),
        'detections': detections,
        'count': len(detections)
    }

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup logger
    logger = setup_logger(
        log_file=config['logging']['log_file'],
        level=config['logging']['level']
    )
    
    logger.info(f"Loading RT-DETR model from {args.model}...")
    model = RTDETR(args.model)
    
    logger.info(f"Running inference on {args.image}...")
    try:
        results = run_inference(
            model=model,
            image_path=args.image,
            imgsz=args.imgsz,
            conf=args.conf,
            save=args.save
        )
        
        logger.info(f"Successfully detected {results['count']} disaster indicators.")
        for det in results['detections']:
            logger.info(f" - Found {det['label']} with {det['confidence']:.2f} confidence.")
            
        if args.save:
            logger.info("Visualization results saved in experiments/inference/disaster_detection")
            
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")

if __name__ == "__main__":
    main()
