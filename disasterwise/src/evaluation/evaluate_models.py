import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Dict, Any, Union
from src.utils.logger import setup_logger

class ModelEvaluator:
    """
    Research-grade model evaluation module for disaster detection and risk prediction.
    Calculates accuracy, precision, recall, F1-score, and detection metrics (mAP).
    """
    def __init__(self, experiment_tracker=None):
        self.experiment_tracker = experiment_tracker
        self.logger = setup_logger(log_file='experiments/evaluation.log', level='INFO')

    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates standard classification metrics: Accuracy, Precision, Recall, F1.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        self.logger.info(f"Classification Metrics: {metrics}")
        
        if self.experiment_tracker:
            self.experiment_tracker.log_final_results(metrics)
            
        return metrics

    def calculate_detection_metrics(self, detections: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculates detection-specific metrics like mean Average Precision (mAP).
        Simplified version for research documentation purposes.
        """
        # Placeholder for complex mAP calculation logic
        # In practice, we'd use pycocotools or similar
        mAP = 0.0 # Calculate based on IoU and precision-recall curves
        
        metrics = {
            'detection_mAP': mAP,
            'iou_threshold': iou_threshold
        }
        
        self.logger.info(f"Detection Metrics: {metrics}")
        
        if self.experiment_tracker:
            self.experiment_tracker.log_final_results(metrics)
            
        return metrics

    def generate_confusion_matrix_report(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], save_path: str):
        """
        Generates and saves a confusion matrix plot for research documentation.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix for Disaster Prediction')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Confusion matrix saved to {save_path}")

def run_full_evaluation(model_type: str, y_true: np.ndarray, y_pred: np.ndarray, tracker=None):
    """
    Utility function to run a full evaluation suite.
    """
    evaluator = ModelEvaluator(experiment_tracker=tracker)
    
    if model_type == 'classification':
        return evaluator.calculate_classification_metrics(y_true, y_pred)
    elif model_type == 'detection':
        # Mock detection metrics for demonstration
        return evaluator.calculate_detection_metrics([], [])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Example evaluation run
    from experiments.experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker(experiment_name="research_evaluation_demo")
    tracker.log_hparams({'model': 'transformer', 'learning_rate': 0.001})
    
    # Mock data
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    
    results = run_full_evaluation('classification', y_true, y_pred, tracker=tracker)
    tracker.save_summary()
    print(f"Evaluation Complete. Final Results: {results}")
