import os
import yaml
import argparse
import torch
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.prediction.transformer_model import DisasterRiskTransformer
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Transformer for Disaster Risk Prediction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Transformer model (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to test data features (.npy)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    logger = setup_logger(
        log_file=config['logging']['log_file'],
        level=config['logging']['level']
    )
    
    logger.info("Evaluating prediction model...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = config['model']['prediction']['input_dim']
    model = DisasterRiskTransformer(input_dim=input_dim)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    # Load data (mocking labels for demonstration)
    X_test = np.load(args.data)
    y_test = np.random.randint(0, 2, (X_test.shape[0], 1))
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=config['model']['prediction']['batch_size'])
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.permute(1, 0, 2).to(device)
            outputs = model(batch_X)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    preds_binary = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)
    roc_auc = roc_auc_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, preds_binary)
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    logger.info(f"Prediction Metrics: {json.dumps(metrics, indent=2)}")
    
    # Save metrics
    output_dir = "experiments/metrics/prediction_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "prediction_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    logger.info(f"Prediction evaluation results saved to {output_dir}")

if __name__ == "__main__":
    main()
