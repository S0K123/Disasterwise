import os
import json
import time
import pandas as pd
from typing import Dict, Any, Optional

class ExperimentTracker:
    """
    A research-grade experiment tracker to log hyperparameters, 
    track metrics over time, and store final evaluation results.
    Results are saved in JSON and CSV formats for easy research paper integration.
    """
    def __init__(self, experiment_name: str, base_dir: str = 'experiments/runs'):
        self.experiment_name = experiment_name
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_id = f"{experiment_name}_{self.timestamp}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.hparams = {}
        self.metrics_history = []
        self.final_results = {}

    def log_hparams(self, hparams: Dict[str, Any]):
        """
        Logs hyperparameters for the current run.
        """
        self.hparams.update(hparams)
        hparams_path = os.path.join(self.run_dir, 'hparams.json')
        with open(hparams_path, 'w') as f:
            json.dump(self.hparams, f, indent=4)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Logs metrics for a specific epoch.
        """
        metrics_entry = {'epoch': epoch, 'timestamp': time.time()}
        metrics_entry.update(metrics)
        self.metrics_history.append(metrics_entry)
        
        # Save history to CSV
        history_path = os.path.join(self.run_dir, 'metrics_history.csv')
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(history_path, index=False)

    def log_final_results(self, results: Dict[str, float]):
        """
        Logs the final evaluation results (e.g., mAP, F1, Accuracy).
        """
        self.final_results.update(results)
        results_path = os.path.join(self.run_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.final_results, f, indent=4)

    def save_summary(self):
        """
        Generates a summary file for the experiment run.
        """
        summary = {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'hparams': self.hparams,
            'final_results': self.final_results
        }
        summary_path = os.path.join(self.run_dir, 'run_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        return summary_path

def compare_experiments(runs_dir: str = 'experiments/runs', output_path: str = 'experiments/comparison_report.csv'):
    """
    Aggregates results from all runs into a single comparison report.
    """
    all_summaries = []
    if not os.path.exists(runs_dir):
        return None

    for run_folder in os.listdir(runs_dir):
        summary_path = os.path.join(runs_dir, run_folder, 'run_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                
                # Flatten summary for DataFrame
                flat_summary = {
                    'run_id': summary['run_id'],
                    'experiment_name': summary['experiment_name'],
                    'timestamp': summary['timestamp']
                }
                flat_summary.update(summary['hparams'])
                flat_summary.update(summary['final_results'])
                all_summaries.append(flat_summary)
    
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        df.to_csv(output_path, index=False)
        return df
    return None
