import torch
import numpy as np
from typing import Dict

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(model, data, n_samples=25):
    """
    Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes with dropout enabled.
    """
    predictions = []
    model.eval()  # Set model to evaluation mode
    enable_dropout(model) # Re-enable dropout layers
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(data)
            predictions.append(output.cpu().numpy())
            
    predictions = np.array(predictions)
    
    # Compute mean and variance
    mean_prediction = np.mean(predictions, axis=0)
    variance = np.var(predictions, axis=0)
    
    return mean_prediction, variance

def estimate_uncertainty(model, data, n_samples=25) -> Dict:
    """
    Estimates prediction uncertainty using Monte Carlo Dropout.
    """
    mean_preds, variance = get_monte_carlo_predictions(model, data, n_samples)
    
    # For now, we'll use a single prediction output and assign it to a generic risk
    # In a real scenario, the model would be multi-headed for different disaster types
    probability = float(mean_preds[0][0])
    uncertainty_score = float(variance[0][0])
    confidence_score = 1 - uncertainty_score

    return {
        "flood_risk": probability, # Placeholder
        "cyclone_risk": probability, # Placeholder
        "earthquake_risk": probability, # Placeholder
        "confidence_score": confidence_score,
        "uncertainty_score": uncertainty_score
    }
