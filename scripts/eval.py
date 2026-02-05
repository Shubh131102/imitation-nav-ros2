"""
Evaluation script for trained navigation models.

Computes comprehensive metrics including MSE, R2, MAE, and per-velocity statistics.
Supports evaluation on multiple datasets and environments.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))
from data_prep import load_dataset, split_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained model with predict() method
        X_test: Test features (n_samples, n_features)
        y_test: Test labels (n_samples, 2) - [linear_vel, angular_vel]
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Overall metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Per-velocity metrics
    mse_linear = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_angular = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    
    r2_linear = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_angular = r2_score(y_test[:, 1], y_pred[:, 1])
    
    # Prediction statistics
    pred_stats = {
        "linear_vel": {
            "mean": float(y_pred[:, 0].mean()),
            "std": float(y_pred[:, 0].std()),
            "min": float(y_pred[:, 0].min()),
            "max": float(y_pred[:, 0].max())
        },
        "angular_vel": {
            "mean": float(y_pred[:, 1].mean()),
            "std": float(y_pred[:, 1].std()),
            "min": float(y_pred[:, 1].min()),
            "max": float(y_pred[:, 1].max())
        }
    }
    
    metrics = {
        "overall": {
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2)
        },
        "linear_velocity": {
            "mse": float(mse_linear),
            "r2": float(r2_linear)
        },
        "angular_velocity": {
            "mse": float(mse_angular),
            "r2": float(r2_angular)
        },
        "prediction_stats": pred_stats,
        "n_samples": int(len(y_test))
    }
    
    return metrics


def print_metrics(metrics: dict):
    """Print evaluation metrics in readable format."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTest Samples: {metrics['n_samples']}")
    print(f"\nOverall Performance:")
    print(f"  MSE: {metrics['overall']['mse']:.6f}")
    print(f"  MAE: {metrics['overall']['mae']:.6f}")
    print(f"  R²:  {metrics['overall']['r2']:.4f}")
    
    print(f"\nLinear Velocity:")
    print(f"  MSE: {metrics['linear_velocity']['mse']:.6f}")
    print(f"  R²:  {metrics['linear_velocity']['r2']:.4f}")
    
    print(f"\nAngular Velocity:")
    print(f"  MSE: {metrics['angular_velocity']['mse']:.6f}")
    print(f"  R²:  {metrics['angular_velocity']['r2']:.4f}")
    
    print(f"\nPrediction Statistics:")
    lin = metrics['prediction_stats']['linear_vel']
    ang = metrics['prediction_stats']['angular_vel']
    print(f"  Linear:  mean={lin['mean']:.3f}, std={lin['std']:.3f}, range=[{lin['min']:.3f}, {lin['max']:.3f}]")
    print(f"  Angular: mean={ang['mean']:.3f}, std={ang['std']:.3f}, range=[{ang['min']:.3f}, {ang['max']:.3f}]")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained navigation model")
    parser.add_argument("--model", type=str, default="models/bc_mlp.joblib",
                       help="Path to trained model file")
    parser.add_argument("--data", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save evaluation results (JSON)")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Test set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load and split data
    logger.info(f"Loading data from {args.data}")
    X, y = load_dataset(args.data)
    _, X_test, _, y_test = split_dataset(X, y, test_ratio=args.test_ratio, random_seed=args.seed)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print results
    print_metrics(metrics)
    
    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metrics["model_path"] = str(model_path)
        metrics["data_path"] = args.data
        metrics["test_ratio"] = args.test_ratio
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
