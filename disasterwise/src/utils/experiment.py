import os
import yaml
import json
import shutil
from datetime import datetime

class ExperimentTracker:
    """
    Manages experiment tracking, including directory creation, artifact saving, and metric logging.
    """
    def __init__(self, config: dict, experiment_name: str):
        self.config = config
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("experiments", f"{self.experiment_name}_{self.timestamp}")
        self._create_experiment_dir()

    def _create_experiment_dir(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        # Save the config for reproducibility
        with open(os.path.join(self.exp_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.config, f)

    def log_metrics(self, metrics: dict):
        with open(os.path.join(self.exp_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)

    def get_log_file_path(self):
        return os.path.join(self.exp_dir, "training_log.txt")

    def save_checkpoint(self, model_path: str):
        shutil.copy(model_path, os.path.join(self.exp_dir, "model_checkpoint.pt"))
