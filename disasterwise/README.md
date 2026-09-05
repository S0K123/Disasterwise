# Disasterwise

A professional research-grade Python project for disaster detection and prediction using satellite imagery and environmental data.

## Project Structure

```
disasterwise/
│
├── configs/            # Configuration files (YAML)
│   └── config.yaml
│
├── data/               # Data storage
│   ├── raw/            # Original data (e.g., xview, satellite imagery)
│   ├── processed/      # Cleaned and processed data
│   └── splits/         # Train/Validation/Test splits
│
├── models/             # Model architectures
│   ├── detection/      # Object detection models for disaster assessment
│   └── prediction/     # Time-series or spatial prediction models
│
├── src/                # Source code
│   ├── data_loader/    # Data loading and batching
│   ├── preprocessing/  # Image and data preprocessing
│   ├── training/       # Training scripts and loss functions
│   ├── inference/      # Prediction and model deployment
│   ├── evaluation/     # Metrics and performance evaluation
│   └── utils/          # Logging, YAML parsing, and helpers
│
├── notebooks/          # Exploratory Data Analysis (EDA) and prototyping
│
├── experiments/        # Experiment logs and results
│
├── saved_models/       # Trained model checkpoints
│
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
└── main.py             # Entry point for the project
```

## Features
- **Clean Architecture**: Modular design for easy integration.
- **Configuration-Driven**: All parameters managed via YAML.
- **Logging**: Comprehensive logging for experiment tracking.
- **Research-Grade**: Suitable for university-level research projects.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Explore the pipeline:
   The project uses a subcommand-based CLI. You can run individual stages or the full pipeline:
   
   - **Preprocess**: `python main.py preprocess`
    - **Tile**: `python main.py tile --tile_size 512 --overlap 50`
    - **Train Detection**: `python main.py train-detection --model rtdetr-l.pt`
   - **Train Prediction**: `python main.py train-prediction --epochs 50`
   - **Full Inference**: `python main.py inference --image path/to/img.png --det_model weights/det.pt --pred_model weights/pred.pt`
   - **Explainability**: `python main.py explain --model weights/pred.pt --data data/processed/features.npy`

3. Sample Execution:
   For a quick overview, run the sample pipeline script:
   ```bash
   python run_pipeline.py
   ```
