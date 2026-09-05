import os
import argparse
import yaml
import logging

def setup_args():
    parser = argparse.ArgumentParser(description="Disasterwise: Professional Disaster Detection and Prediction Project")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Preprocess subcommand
    preprocess_parser = subparsers.add_parser("preprocess", help="Run data preprocessing and splitting")
    preprocess_parser.add_argument("--test_size", type=float, help="Test set ratio")
    preprocess_parser.add_argument("--val_size", type=float, help="Validation set ratio")

    # Tile subcommand
    tile_parser = subparsers.add_parser("tile", help="Split satellite images and annotations into patches")
    tile_parser.add_argument("--tile_size", type=int, help="Size of each tile (e.g., 256 or 512)")
    tile_parser.add_argument("--overlap", type=int, help="Overlap between tiles in pixels")

    # Train Detection subcommand
    train_det_parser = subparsers.add_parser("train-detection", help="Train RT-DETR detection model")
    train_det_parser.add_argument("--model", type=str, help="Pre-trained weights")
    train_det_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_det_parser.add_argument("--batch", type=int, help="Batch size")

    # Train Prediction subcommand
    train_pred_parser = subparsers.add_parser("train-prediction", help="Train Transformer risk prediction model")
    train_pred_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_pred_parser.add_argument("--input_dim", type=int, help="Input feature dimension")
    train_pred_parser.add_argument("--batch", type=int, help="Batch size")

    # Inference subcommand
    inference_parser = subparsers.add_parser("inference", help="Run end-to-end inference")
    inference_parser.add_argument("--pre_image", type=str, required=True, help="Path to pre-disaster satellite image")
    inference_parser.add_argument("--post_image", type=str, required=True, help="Path to post-disaster satellite image")
    inference_parser.add_argument("--det_model", type=str, required=True, help="Path to detection model weights")
    inference_parser.add_argument("--pred_model", type=str, required=True, help="Path to prediction model weights")

    # Explain subcommand
    explain_parser = subparsers.add_parser("explain", help="Run Explainable AI (XAI) analysis")
    explain_parser.add_argument("--model", type=str, required=True, help="Path to prediction model")
    explain_parser.add_argument("--data", type=str, required=True, help="Path to data for explanation")

    # Evaluate Detection subcommand
    eval_det_parser = subparsers.add_parser("evaluate-detection", help="Evaluate RT-DETR detection model")
    eval_det_parser.add_argument("--model", type=str, required=True, help="Path to trained RT-DETR model (.pt)")
    eval_det_parser.add_argument("--data_split", type=str, default="test", help="Data split to evaluate on (val or test)")

    # Evaluate Prediction subcommand
    eval_pred_parser = subparsers.add_parser("evaluate-prediction", help="Evaluate Transformer risk prediction model")
    eval_pred_parser.add_argument("--model", type=str, required=True, help="Path to trained Transformer model (.pt)")
    eval_pred_parser.add_argument("--data", type=str, required=True, help="Path to test data features (.npy)")

    # Temporal Analysis subcommand
    temporal_parser = subparsers.add_parser("temporal-analysis", help="Run temporal analysis between two images")
    temporal_parser.add_argument("--pre_image", type=str, required=True, help="Path to pre-disaster satellite image")
    temporal_parser.add_argument("--post_image", type=str, required=True, help="Path to post-disaster satellite image")
    temporal_parser.add_argument("--det_model", type=str, required=True, help="Path to detection model weights")

    return parser.parse_args()

def run_full_inference(args, config, logger):
    """
    Executes the complete pipeline: Detection -> Feature Extraction -> Risk Prediction.
    """
    import torch
    import numpy as np
    from ultralytics import RTDETR
    from src.inference.detect_disaster import run_inference
    from src.preprocessing.feature_extraction import FeatureExtractor
    from models.prediction.transformer_model import DisasterRiskTransformer
    from src.evaluation.uncertainty_estimation import estimate_uncertainty
    from src.preprocessing.temporal_analysis import (
        load_pre_post_images, align_images, compute_change_features
    )
    from src.data_loader.environmental_data_loader import EnvironmentalDataLoader

    logger.info("Starting end-to-end inference pipeline...")
    
    # 1. Temporal Analysis
    logger.info("Step 1: Performing temporal analysis...")
    pre_image, post_image = load_pre_post_images(args.pre_image, args.post_image)
    pre_image, post_image = align_images(pre_image, post_image)

    # 2. Detection on both images
    logger.info("Step 2: Running detection on pre- and post-disaster images...")
    try:
        det_model = RTDETR(args.det_model)
        pre_detections = run_inference(model=det_model, image_path=args.pre_image)
        post_detections = run_inference(model=det_model, image_path=args.post_image)
        all_detections = {"pre": pre_detections['detections'], "post": post_detections['detections']}
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        return

    # 3. Feature Extraction
    logger.info("Step 3: Extracting features...")
    try:
        # Temporal features
        change_features = compute_change_features(pre_image, post_image, all_detections)
        logger.info(f"Temporal change features: {change_features}")

        # Environmental features
        env_loader = EnvironmentalDataLoader(config)
        # This is a placeholder for real-time data alignment
        env_data = env_loader.get_data_for_timestamp(env_loader.load_from_csv('path/to/your/env_data.csv'), pd.Timestamp.now())

        # Combine features
        extractor = FeatureExtractor(config)
        feature_vector = extractor.detections_to_vector(
            post_detections['detections'], 
            env_data, 
            change_features
        )
        logger.info(f"Combined feature vector extracted (dim={len(feature_vector)}).")

    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return

    # 3. Risk Prediction Stage
    logger.info("Step 3: Predicting disaster risk probability...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = config['model']['prediction'].get('input_dim', 20)
        d_model = config['model']['prediction'].get('d_model', 64)
        
        pred_model = DisasterRiskTransformer(input_dim=input_dim, d_model=d_model).to(device)
        pred_model.load_state_dict(torch.load(args.pred_model, map_location=device))
        pred_model.eval()
        
        # Prepare input for Transformer [seq_len, batch_size, input_dim]
        features_tensor = torch.from_numpy(feature_vector).unsqueeze(0).unsqueeze(1).to(device)
        
        with torch.no_grad():
            risk_prob = pred_model(features_tensor).item()
            
        logger.info(f"FINAL PREDICTION: Disaster Risk Probability = {risk_prob:.4f}")
        
        # 4. Uncertainty Estimation
        logger.info("Step 4: Estimating prediction uncertainty...")
        uncertainty_results = estimate_uncertainty(pred_model, features_tensor)
        logger.info(f"Uncertainty Estimation: {uncertainty_results}")
        
        if risk_prob > 0.8:
            logger.warning("ALERT: HIGH DISASTER RISK DETECTED!")
        elif risk_prob > 0.5:
            logger.info("ADVISORY: Moderate disaster risk detected. Monitor area.")
        else:
            logger.info("STATUS: Low risk. Area appears stable.")
            
    except Exception as e:
        logger.error(f"Risk prediction failed: {str(e)}")

def run_temporal_analysis(args, config, logger):
    """
    Runs the temporal analysis pipeline.
    """
    from ultralytics import RTDETR
    from src.inference.detect_disaster import run_inference
    from src.preprocessing.temporal_analysis import (
        load_pre_post_images, align_images, compute_change_features
    )

    logger.info("Starting temporal analysis...")
    pre_image, post_image = load_pre_post_images(args.pre_image, args.post_image)
    pre_image, post_image = align_images(pre_image, post_image)

    det_model = RTDETR(args.det_model)
    pre_detections = run_inference(model=det_model, image_path=args.pre_image)
    post_detections = run_inference(model=det_model, image_path=args.post_image)
    all_detections = {"pre": pre_detections['detections'], "post": post_detections['detections']}

    change_features = compute_change_features(pre_image, post_image, all_detections)
    logger.info(f"Temporal change features: {change_features}")

def main():
    args = setup_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    from src.utils.logger import setup_logger
    logger = setup_logger(
        log_file=config['logging']['log_file'],
        level=config['logging']['level']
    )

    if args.command == "preprocess":
        from src.preprocessing.preprocess_dataset import main as preprocess_main
        preprocess_main()
    elif args.command == "tile":
        from src.preprocessing.image_tiling import main as tiling_main
        tiling_main()
    elif args.command == "train-detection":
        from src.training.train_detection import main as train_detection_main
        train_detection_main()
    elif args.command == "train-prediction":
        from src.training.train_prediction import main as train_prediction_main
        train_prediction_main()
    elif args.command == "inference":
        run_full_inference(args, config, logger)
    elif args.command == "explain":
        from src.evaluation.explain_model import main as explain_model_main
        explain_model_main()
    elif args.command == "evaluate-detection":
        from src.evaluation.evaluate_detection import main as evaluate_detection_main
        evaluate_detection_main()
    elif args.command == "evaluate-prediction":
        from src.evaluation.evaluate_prediction import main as evaluate_prediction_main
        evaluate_prediction_main()
    elif args.command == "temporal-analysis":
        run_temporal_analysis(args, config, logger)
    else:
        print("Usage: python main.py [preprocess|tile|train-detection|train-prediction|inference|explain|evaluate-detection|evaluate-prediction|temporal-analysis] --help")

if __name__ == "__main__":
    main()
