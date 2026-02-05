"""
Neural network models for imitation learning navigation.

Implements behavioral cloning policy with support for Monte Carlo Dropout
uncertainty estimation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BCPolicy(nn.Module):
    """
    Behavioral Cloning Policy Network.
    
    Fully-connected neural network that maps LiDAR scans to velocity commands.
    Supports both deterministic inference and MC Dropout for uncertainty estimation.
    
    Architecture:
        Input(360) -> Linear(256) -> ReLU -> Dropout ->
        Linear(256) -> ReLU -> Dropout ->
        Linear(2) -> Output [linear_vel, angular_vel]
    
    Args:
        in_dim: Input dimension (default: 360 for full LiDAR scan)
        hidden_dim: Hidden layer dimension (default: 256)
        out_dim: Output dimension (default: 2 for linear and angular velocity)
        dropout_rate: Dropout probability for MC Dropout (default: 0.2)
    """
    
    def __init__(
        self,
        in_dim: int = 360,
        hidden_dim: int = 256,
        out_dim: int = 2,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        
        # Network layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input LiDAR scan (batch_size, in_dim)
            
        Returns:
            Velocity commands (batch_size, out_dim) - [linear_vel, angular_vel]
        """
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def predict(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Predict velocity commands.
        
        Args:
            x: Input LiDAR scan (batch_size, in_dim)
            deterministic: If True, disable dropout for deterministic prediction
            
        Returns:
            Velocity commands (batch_size, out_dim)
        """
        if deterministic:
            self.eval()
        else:
            self.train()  # Keep dropout active for stochastic predictions
        
        with torch.no_grad():
            return self.forward(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout.
        
        Performs multiple stochastic forward passes with dropout enabled
        to estimate epistemic uncertainty.
        
        Args:
            x: Input LiDAR scan (batch_size, in_dim)
            n_samples: Number of MC forward passes (default: 20)
            
        Returns:
            mean_prediction: Mean velocity command (batch_size, out_dim)
            uncertainty: Standard deviation across predictions (batch_size, out_dim)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        # Stack predictions (n_samples, batch_size, out_dim)
        predictions = torch.stack(predictions)
        
        # Compute mean and std
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_prediction, uncertainty
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """Save model weights to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'in_dim': self.in_dim,
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'dropout_rate': self.dropout_rate
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model on ('cpu' or 'cuda')
            
        Returns:
            Loaded BCPolicy model
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            in_dim=checkpoint['in_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            out_dim=checkpoint['out_dim'],
            dropout_rate=checkpoint['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        return model


class Conv1DPolicy(nn.Module):
    """
    Conv1D-based Behavioral Cloning Policy.
    
    Uses 1D convolutions to process sequential LiDAR data, preserving
    spatial relationships between nearby measurements.
    
    Architecture:
        Input(360) -> Conv1D(16, k=5) -> ReLU -> MaxPool(2) -> Dropout ->
        Conv1D(32, k=5) -> ReLU -> MaxPool(2) -> Dropout ->
        Flatten -> Linear(64) -> ReLU -> Dropout ->
        Linear(2) -> Output
    
    Args:
        in_channels: Input channels (default: 1)
        hidden_dim: Fully connected hidden dimension (default: 64)
        out_dim: Output dimension (default: 2)
        dropout_rate: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        out_dim: int = 2,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate flattened size: 360 -> /2 -> /2 = 90, with 32 channels
        self.flat_size = 32 * 90
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input LiDAR (batch_size, 360)
            
        Returns:
            Velocity commands (batch_size, 2)
        """
        # Add channel dimension (batch_size, 1, 360)
        x = x.unsqueeze(1)
        
        # Conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc_layers(x)
        
        return x


if __name__ == "__main__":
    # Test BCPolicy
    print("Testing BCPolicy...")
    model = BCPolicy(in_dim=360, hidden_dim=256, out_dim=2)
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    x = torch.randn(4, 360)  # Batch of 4 samples
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    
    # Test MC Dropout uncertainty
    mean, uncertainty = model.predict_with_uncertainty(x, n_samples=20)
    print(f"Mean: {mean.shape}, Uncertainty: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean(dim=1)}")
    
    # Test save/load
    model.save('/tmp/test_model.pth')
    loaded_model = BCPolicy.load('/tmp/test_model.pth')
    print("Save/load successful")
    
    # Test Conv1DPolicy
    print("\nTesting Conv1DPolicy...")
    conv_model = Conv1DPolicy()
    y_conv = conv_model(x)
    print(f"Conv1D output shape: {y_conv.shape}")
