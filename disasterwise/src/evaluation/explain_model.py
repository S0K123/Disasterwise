import os
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from models.prediction.transformer_model import DisasterRiskTransformer
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Explainable AI for Disaster Prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Transformer model (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to sample data for explanation (.npy)")
    parser.add_argument("--feature_names", type=str, help="Comma-separated list of feature names")
    return parser.parse_args()

class DisasterExplainer:
    """
    Handles Explainable AI (XAI) for the disaster prediction model using SHAP.
    """
    def __init__(self, model: torch.nn.Module, feature_names: List[str], device: str = 'cpu'):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.eval()
        self.model.to(device)

    def model_wrapper(self, x_np: np.ndarray) -> np.ndarray:
        """
        Wrapper to make the PyTorch model compatible with SHAP's numpy interface.
        SHAP expects (batch, features), but Transformer expects (seq_len, batch, features).
        We assume SHAP provides a batch of 'flattened' sequences or single steps.
        For simplicity in this research-grade tool, we explain the last time step's influence.
        """
        # Convert numpy to torch
        # Note: If x_np is (batch, input_dim), we treat it as a single-step sequence
        # or expand it to match the model's expected sequence length.
        x_torch = torch.from_numpy(x_np).float().to(self.device)
        
        # If input is (batch, features), add seq_len dimension: (1, batch, features)
        if len(x_torch.shape) == 2:
            x_torch = x_torch.unsqueeze(0)
            
        with torch.no_grad():
            preds = self.model(x_torch)
        return preds.cpu().numpy()

    def explain_global(self, background_data: np.ndarray, test_data: np.ndarray, save_path: str):
        """
        Generates global feature importance using SHAP Summary Plot.
        """
        print("Generating global SHAP explanations...")
        explainer = shap.KernelExplainer(self.model_wrapper, background_data)
        shap_values = explainer.shap_values(test_data)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_data, feature_names=self.feature_names, show=False)
        plt.title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "shap_summary_plot.png"))
        plt.close()
        print(f"Global summary plot saved to {save_path}")

    def explain_local(self, instance: np.ndarray, background_data: np.ndarray, save_path: str, index: int = 0):
        """
        Generates local explanation for a single prediction using SHAP Force Plot.
        """
        print(f"Generating local SHAP explanation for instance {index}...")
        explainer = shap.KernelExplainer(self.model_wrapper, background_data)
        shap_values = explainer.shap_values(instance)
        
        # Waterfall plot for local attribution
        plt.figure(figsize=(12, 4))
        # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], feature_names=self.feature_names)
        # Using a simpler bar plot for research report compatibility
        plt.barh(self.feature_names, shap_values[0])
        plt.title(f"Local Feature Attribution for Instance {index}")
        plt.xlabel("SHAP Value (Impact on Prediction)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"local_explanation_{index}.png"))
        plt.close()
        print(f"Local explanation saved to {save_path}")

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    logger = setup_logger(log_file=config['logging']['log_file'], level=config['logging']['level'])
    
    # Feature names setup
    if args.feature_names:
        feature_names = args.feature_names.split(',')
    else:
        # Default research-grade feature names for disasterwise
        feature_names = [
            'temp', 'humidity', 'wind_speed', 'rainfall',
            'building_count', 'damage_severity', 'flood_ratio', 'debris_density',
            'infra_integrity', 'vegetation_index', 'soil_moisture', 'elevation',
            'pop_density', 'dist_to_coast', 'historical_risk', 'slope',
            'land_cover', 'pressure', 'cloud_cover', 'visibility'
        ]

    # Model and Data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = config['model']['prediction']['input_dim']
    model = DisasterRiskTransformer(input_dim=input_dim)
    model.load_state_dict(torch.load(args.model, map_location=device))
    
    # Load data
    if os.path.exists(args.data):
        data = np.load(args.data)
    else:
        logger.warning(f"Data file {args.data} not found. Using mock data for demonstration.")
        data = np.random.randn(50, input_dim) # 50 samples, 20 features

    # Prepare directories
    output_dir = os.path.join(config['experiments'].get('output_dir', 'experiments'), 'xai_reports')
    os.makedirs(output_dir, exist_ok=True)

    # Explainer initialization
    explainer = DisasterExplainer(model, feature_names, device=device)
    
    # SHAP requires a background dataset for reference (e.g., first 20 samples)
    background = data[:20]
    test_samples = data[20:40]
    
    # Run explanations
    explainer.explain_global(background, test_samples, output_dir)
    explainer.explain_local(data[40:41], background, output_dir, index=40)
    
    logger.info(f"XAI Analysis complete. Reports and plots generated in {output_dir}")

if __name__ == "__main__":
    main()
