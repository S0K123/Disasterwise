import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, List
from models.prediction.transformer_model import DisasterRiskTransformer
from src.utils.logger import setup_logger
from src.utils.experiment import ExperimentTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer for Disaster Risk Prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch", type=int, help="Batch size (overrides config)")
    parser.add_argument("--input_dim", type=int, default=20, help="Input feature dimension")
    return parser.parse_args()

def prepare_mock_data(num_samples: int = 1000, seq_len: int = 10, input_dim: int = 20):
    """
    Creates mock time-series data for training.
    X: [num_samples, seq_len, input_dim]
    y: [num_samples, 1]
    """
    X = torch.randn(num_samples, seq_len, input_dim)
    y = torch.randint(0, 2, (num_samples, 1)).float()
    return X, y

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Experiment Tracking
    tracker = ExperimentTracker(config, experiment_name="transformer_risk_prediction")
    
    logger = setup_logger(
        log_file=tracker.get_log_file_path(),
        level=config['logging']['level']
    )
    
    logger.info("Initializing Transformer Training Pipeline for Risk Prediction...")
    
    # Configuration
    epochs = args.epochs or config['model']['prediction'].get('epochs', 50)
    batch_size = args.batch or config['model']['prediction'].get('batch_size', 16)
    input_dim = args.input_dim or config['model']['prediction'].get('input_dim', 20)
    d_model = config['model']['prediction'].get('d_model', 64)
    seq_len = config['model']['prediction'].get('seq_len', 10)
    
    # Data Preparation (Replace with actual data loading logic)
    X, y = prepare_mock_data(num_samples=1000, seq_len=seq_len, input_dim=input_dim)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterRiskTransformer(input_dim=input_dim, d_model=d_model).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy for probability prediction
    
    # Training Loop
    logger.info(f"Starting training on {device}...")
    model.train()
    training_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            # Transformer expects [seq_len, batch_size, input_dim]
            batch_X = batch_X.permute(1, 0, 2).to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        training_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
    # Log metrics
    tracker.log_metrics({
        'final_training_loss': training_losses[-1],
        'all_training_losses': training_losses
    })
    
    # Save checkpoint
    save_path = os.path.join(tracker.exp_dir, 'model_checkpoint.pt')
    torch.save(model.state_dict(), save_path)
    logger.info(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
