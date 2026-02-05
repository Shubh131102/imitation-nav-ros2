"""
Training script for behavioral cloning navigation model.

Trains MLP regressor on expert demonstrations with configurable architecture
and hyperparameters. Supports checkpointing and training visualization.
"""

import argparse
import json
import logging
import time
from pathlib import Path
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))
from data_prep import load_dataset, split_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_training_history(train_losses: list, output_path: Path):
    """
    Plot training loss curve.
    
    Args:
        train_losses: List of loss values per iteration
        output_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Training curve saved to {output_path}")


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                hidden_layers: tuple = (128, 64),
                max_iter: int = 500,
                learning_rate: float = 0.001,
                random_seed: int = 42) -> tuple:
    """
    Train MLP regressor for navigation.
    
    Args:
        X_train: Training features (n_samples, 360)
        y_train: Training labels (n_samples, 2)
        X_test: Test features
        y_test: Test labels
        hidden_layers: Tuple of hidden layer sizes
        max_iter: Maximum training iterations
        learning_rate: Initial learning rate for Adam
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    logger.info(f"Training MLP with architecture: Input(360) -> {hidden_layers} -> Output(2)")
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize model
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=random_seed,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Training history
    history = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "n_iterations": model.n_iter_,
        "train_time_seconds": float(train_time),
        "loss_curve": model.loss_curve_,
        "architecture": {
            "hidden_layers": hidden_layers,
            "n_parameters": sum([w.size for w in model.coefs_]) + sum([b.size for b in model.intercepts_])
        }
    }
    
    logger.info(f"\nTraining completed in {train_time:.2f}s ({model.n_iter_} iterations)")
    logger.info(f"Train MSE: {train_mse:.6f} | R²: {train_r2:.4f}")
    logger.info(f"Test MSE:  {test_mse:.6f} | R²: {test_r2:.4f}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train behavioral cloning navigation model")
    parser.add_argument("--data", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, default="models/bc_mlp.joblib",
                       help="Path to save trained model")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64],
                       help="Hidden layer sizes (e.g., --hidden-layers 128 64)")
    parser.add_argument("--max-iter", type=int, default=500,
                       help="Maximum training iterations")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Initial learning rate")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--plot", action="store_true",
                       help="Generate training curve plot")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_ratio=args.test_ratio, random_seed=args.seed
    )
    
    # Train model
    model, history = train_model(
        X_train, y_train, X_test, y_test,
        hidden_layers=tuple(args.hidden_layers),
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        random_seed=args.seed
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")
    
    # Save training history
    history_path = output_path.with_suffix('.json')
    with open(history_path, 'w') as f:
        # Convert loss curve to list for JSON serialization
        history_copy = history.copy()
        history_copy['loss_curve'] = [float(x) for x in history['loss_curve']]
        json.dump(history_copy, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Plot training curve
    if args.plot:
        plot_path = output_path.parent / f"{output_path.stem}_loss_curve.png"
        plot_training_history(history['loss_curve'], plot_path)


if __name__ == "__main__":
    main()
