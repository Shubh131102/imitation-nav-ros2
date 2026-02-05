"""
Training script for behavioral cloning navigation policy.

Supports Phase 1 (baseline BC) and Phase 2.5 (DAgger) training with
configurable hyperparameters, validation, and checkpoint saving.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from model import BCPolicy, Conv1DPolicy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_npz(path: str, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load dataset from compressed numpy archive.
    
    Args:
        path: Path to .npz file containing 'X' and 'y' arrays
        normalize: If True, normalize LiDAR scans to [0, 1]
        
    Returns:
        X: LiDAR scans (n_samples, 360)
        y: Velocity commands (n_samples, 2)
    """
    logger.info(f"Loading data from {path}")
    
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    
    # Normalize LiDAR scans
    if normalize:
        max_val = X.max()
        if max_val > 0:
            X = X / max_val
            logger.info(f"Normalized LiDAR scans by max value: {max_val:.2f}")
    
    # Convert to tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    logger.info(f"Loaded {len(X)} samples")
    logger.info(f"  X shape: {X.shape}, range: [{X.min():.3f}, {X.max():.3f}]")
    logger.info(f"  y shape: {y.shape}")
    logger.info(f"  Linear vel - mean: {y[:, 0].mean():.3f}, std: {y[:, 0].std():.3f}")
    logger.info(f"  Angular vel - mean: {y[:, 1].mean():.3f}, std: {y[:, 1].std():.3f}")
    
    return X_tensor, y_tensor


def create_dataloaders(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 256,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        X: Input features
        y: Target labels
        batch_size: Batch size for training
        val_split: Fraction of data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader
    """
    dataset = TensorDataset(X, y)
    
    # Split into train/val
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"Train samples: {n_train}, Val samples: {n_val}")
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training dataloader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Average training loss
    """
    model.train()
    epoch_losses = []
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    return np.mean(epoch_losses)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> float:
    """
    Validate model on validation set.
    
    Args:
        model: Neural network model
        val_loader: Validation dataloader
        loss_fn: Loss function
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def train(
    data_path: str = "data/tb3_runs.npz",
    output_path: str = "models/policy.pt",
    architecture: str = "mlp",
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    dropout_rate: float = 0.2,
    device: str = "cpu",
    save_best: bool = True,
    random_seed: int = 42
) -> Dict:
    """
    Train behavioral cloning policy.
    
    Args:
        data_path: Path to .npz dataset
        output_path: Path to save trained model
        architecture: Model architecture ('mlp' or 'conv1d')
        hidden_dim: Hidden layer dimension
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for Adam optimizer
        val_split: Validation set fraction
        dropout_rate: Dropout probability
        device: Device to train on ('cpu' or 'cuda')
        save_best: If True, save best model based on validation loss
        random_seed: Random seed for reproducibility
        
    Returns:
        Training history dictionary
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Load data
    X, y = load_npz(data_path)
    train_loader, val_loader = create_dataloaders(
        X, y, batch_size=batch_size, val_split=val_split, random_seed=random_seed
    )
    
    # Create model
    if architecture == "mlp":
        model = BCPolicy(
            in_dim=X.shape[1],
            hidden_dim=hidden_dim,
            out_dim=2,
            dropout_rate=dropout_rate
        )
    elif architecture == "conv1d":
        model = Conv1DPolicy(
            in_channels=1,
            hidden_dim=hidden_dim,
            out_dim=2,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model = model.to(device)
    logger.info(f"Model: {architecture.upper()}")
    logger.info(f"Parameters: {model.get_num_parameters():,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training loop
    history = {
        "train_losses": [],
        "val_losses": [],
        "best_epoch": 0,
        "best_val_loss": float('inf')
    }
    
    logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info("="*60)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        # Record history
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        
        # Save best model
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            
            if save_best:
                model.save(output_path)
        
        # Log progress
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Best: {history['best_val_loss']:.6f} (Epoch {history['best_epoch']})"
        )
    
    logger.info("="*60)
    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch']}")
    
    # Save final model if not saving best
    if not save_best:
        model.save(output_path)
    
    logger.info(f"Model saved to {output_path}")
    
    # Save training history
    history_path = Path(output_path).with_suffix('.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train behavioral cloning navigation policy")
    parser.add_argument("--data", type=str, default="data/tb3_runs.npz",
                       help="Path to .npz dataset")
    parser.add_argument("--output", type=str, default="models/policy.pt",
                       help="Path to save trained model")
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "conv1d"],
                       help="Model architecture")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden layer dimension")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation set fraction")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to train on (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Train
    train(
        data_path=args.data,
        output_path=args.output,
        architecture=args.arch,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        dropout_rate=args.dropout,
        device=args.device,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
