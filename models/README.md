# Models Directory

Trained neural network models and checkpoints from all phases.

## Structure
```
models/
├── phase1_bc.pth              Baseline behavioral cloning model
├── phase2_bc_dropout.pth      BC model with dropout enabled
├── phase2.5_dagger.pth        Final DAgger-refined model
└── checkpoints/               Training checkpoints
```

## Model Specifications

**Architecture: Conv1D CNN**
- Input: 360-dimensional LiDAR scan
- Parameters: 186,178 trainable
- Output: 2-dimensional velocity command (linear, angular)

**Layer Details:**
```
Conv1D(16 filters, kernel=5) + ReLU + MaxPool(2)
Conv1D(32 filters, kernel=5) + ReLU + MaxPool(2)
Flatten
Dense(64) + ReLU + Dropout(0.2)
Output(2)
```

## Model Performance

| Model | Test Loss (MSE) | Training Samples | Notes |
|-------|-----------------|------------------|-------|
| phase1_bc.pth | 0.010 | 1,113 | Baseline, no uncertainty |
| phase2_bc_dropout.pth | 0.010 | 1,113 | MC Dropout enabled |
| phase2.5_dagger.pth | 0.006 | 2,100 | DAgger refined, best performance |

## Usage

**Load Model:**
```python
import torch
from models.network import NavigationCNN

model = NavigationCNN()
model.load_state_dict(torch.load('models/phase2.5_dagger.pth'))
model.eval()
```

**MC Dropout Inference:**
```python
model.train()  # Keep dropout active
predictions = []
for _ in range(20):
    pred = model(lidar_input)
    predictions.append(pred)
mean_pred = torch.mean(torch.stack(predictions), dim=0)
uncertainty = torch.std(torch.stack(predictions), dim=0)
```

## Notes

Models saved in PyTorch format (.pth). Training conducted on CPU (Intel i7, 16GB RAM). Each training session takes approximately 3 minutes for 100 epochs.

Recommended: Use phase2.5_dagger.pth for deployment as it provides best balance of performance and uncertainty calibration.
