# Disasterwise

Disasterwise is an end-to-end disaster analysis system that combines satellite imagery, damage detection, risk prediction, and alert generation.

## What it includes

- FastAPI backend for image analysis and prediction history
- React frontend for uploading pre- and post-disaster images
- RT-DETR-based object detection and transformer-based risk prediction modules
- Configurable preprocessing, training, inference, evaluation, and experiment tracking pipelines
- SQLite-backed history storage for backend results

## Repository layout

```text
disasterwise/
├── backend/       FastAPI service and database integration
├── configs/       YAML configuration
├── frontend/      React application
├── models/        Detection and prediction model definitions
├── src/           Data loading, preprocessing, training, inference, and evaluation
├── experiments/   Experiment tracking utilities
├── run_pipeline.py
└── main.py        Command-line pipeline entry point
samples/           Example images for testing the application
```

## Quick start

### Backend

```bash
cd disasterwise/backend
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python main.py
```

The API runs at `http://127.0.0.1:5000`.

### Frontend

In a second terminal:

```bash
cd disasterwise/frontend
npm install
npm start
```

The web application runs at `http://localhost:3000`.

Upload a pre-disaster image and a post-disaster image, then select **Analyze Damage** to view the result and alert level.

## Research pipeline

Install the root Python dependencies when using the research pipeline:

```bash
cd disasterwise
pip install -r requirements.txt
python main.py --help
```

Example commands:

```bash
python main.py preprocess
python main.py tile --tile_size 512 --overlap 50
python main.py train-detection --model rtdetr-l.pt
python main.py inference --image path/to/image.png --det_model path/to/det.pt --pred_model path/to/pred.pt
```

## Data and model files

Large datasets, trained weights, local uploads, virtual environments, dependency folders, and generated output are intentionally excluded from Git. Place local data under `disasterwise/data/` and model weights under `disasterwise/saved_models/` or provide their paths through the command-line options.

The included `samples/` directory contains small example images for trying the application without downloading the full dataset.

## Documentation

- [Detailed run guide](disasterwise/RUN_GUIDE.md)
- [Research project README](disasterwise/README.md)
- [Root requirements](disasterwise/requirements.txt)
- [Backend requirements](disasterwise/backend/requirements.txt)