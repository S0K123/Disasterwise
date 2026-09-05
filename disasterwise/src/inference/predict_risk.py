import os
import yaml
import argparse
import torch
import numpy as np
from typing import Dict, Any, List
from models.prediction.transformer_model import DisasterRiskTransformer
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Disaster Risk Probability")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Transformer model (.pt)")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input features (.npy, .csv)")
    parser.add_argument("--input_dim", type=int, default=20, help="Input feature dimension")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    logger = setup_logger(
        log_file=config['logging']['log_file'],
        level=config['logging']['level']
    )
    
    logger.info(f"Loading Transformer model from {args.model}...")
    
    # Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterRiskTransformer(input_dim=args.input_dim, d_model=64).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    logger.info(f"Running inference on {args.input_data}...")
    
    try:
        # Load and prepare input features (assuming .npy format for example)
        # Sequence format: [seq_len, batch_size, input_dim]
        # In this mock, we assume input data is [seq_len, 1, input_dim] for a single sample
        if args.input_data.endswith('.npy'):
            features = np.load(args.input_data)
        else:
            # Mock loading for non-npy files for demonstration
            features = np.random.randn(10, 1, args.input_dim)
            
        features = torch.from_numpy(features).float().to(device)
        
        with torch.no_grad():
            risk_probability = model(features).item()
            
        logger.info(f"Predicted Disaster Risk Probability: {risk_probability:.4f}")
        
        # Actionable alert logic
        if risk_probability > 0.8:
            logger.warning("HIGH RISK DETECTED: Critical disaster risk predicted!")
        elif risk_probability > 0.5:
            logger.info("MODERATE RISK DETECTED: Monitor environmental indicators closely.")
        else:
            logger.info("LOW RISK: No immediate disaster risk predicted.")
            
    except Exception as e:
        logger.error(f"Risk prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
