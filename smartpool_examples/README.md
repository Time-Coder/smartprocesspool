# SmartPool Examples

This package contains practical examples demonstrating the capabilities of SmartPool for various computational tasks.

## Examples Overview

### 1. Prime Number Counting (`count_prime`)

A classic benchmark comparing different parallel processing approaches for CPU-intensive computations.

#### Running the Example

```bash
python run_count_prime.py
```

#### What it Demonstrates

- Performance comparison between ProcessPool, ThreadPool, and sequential execution
- Resource-aware task scheduling based on CPU core availability
- Memory-efficient parallel processing for mathematical computations

#### Sample Output
```
Sequential execution: 12.45s
ThreadPool execution: 8.23s  
ProcessPool execution: 3.15s
Speedup: 3.95x compared to sequential
```

### 2. Cross-Validation with Deep Learning (`cross_validation`)

Demonstrates SmartPool's capabilities for machine learning workloads with GPU resource management.

#### Running the Example

```bash
# Using ThreadPool
python run_cross_validation.py --pool smartpool.ThreadPool

# Using ProcessPool  
python run_cross_validation.py --pool smartpool.ProcessPool

# Using Ray (external comparison)
python run_cross_validation.py --pool ray
```

#### Key Features Demonstrated

- **PyTorch Tensor Sharing**: Efficient tensor transfer between processes without serialization
- **Training Hot Migration**: Automatic movement of CPU training tasks to GPU when available
- **Optimizer Device Management**: Proper handling of optimizer states during device migration
- **Flexible Import System**: Modular design allowing selective import of needed components
- **Dynamic Device Selection**: Using `best_device()` for optimal compute resource utilization
- **GPU Memory Management**: Intelligent allocation and monitoring of GPU resources

#### Supported Models

The example includes implementations of popular deep learning architectures:
- MLP (Multi-Layer Perceptron)
- LeNet5
- ModernCNN
- ResNet variants (ResNet, ResNetV2)
- ResNeXt variants (ResNeXt, ResNeXtV2)

#### What it Demonstrates

- GPU memory management and core allocation
- Automatic device selection (CPU vs GPU)
- Cross-validation pipeline parallelization
- Resource monitoring during training
- Performance comparison with external frameworks

#### Configuration

The example can be configured through `config.py`:

```python
# Key configuration options
DATASET_NAME = "MNIST"          # or "FashionMNIST", "CIFAR10"
MODEL_NAME = "ResNet18"         # Model architecture
N_FOLDS = 5                     # Cross-validation folds
BATCH_SIZE = 128                # Training batch size
MAX_WORKERS = 0                 # 0 = auto-detect
```

## Performance Benchmarks

### System Requirements for Examples

- **Minimum**: 4 CPU cores, 8GB RAM
- **Recommended**: 8+ CPU cores, 16GB+ RAM, CUDA-compatible GPU
- **For Deep Learning**: NVIDIA GPU with 4GB+ VRAM

### Expected Performance Gains

| Task | Sequential | ThreadPool | ProcessPool | Speedup |
|------|------------|------------|-------------|---------|
| Prime Counting (1M numbers) | 12.45s | 8.23s | 3.15s | 3.95x |
| Cross-Validation (5-fold) | 45.2s | 28.7s | 18.3s | 2.47x |

## Running Specific Examples

### Prime Number Counter

```python
# Direct module execution
python -m smartpool_examples.count_prime

# With custom parameters
python -m smartpool_examples.count_prime --start 1 --stop 1000000
```

### Cross-Validation

```python
# Basic usage
python -m smartpool_examples.cross_validation

# Custom configuration
python -m smartpool_examples.cross_validation \
    --dataset CIFAR10 \
    --model ResNet34 \
    --folds 10 \
    --batch-size 64 \
    --workers 8
```

## Example Code Structure

```
smartpool_examples/
├── count_prime/
│   ├── __init__.py
│   └── count_prime.py          # Prime counting functions
├── cross_validation/
│   ├── models/                 # Deep learning model implementations
│   │   ├── MLP.py
│   │   ├── LeNet5.py
│   │   ├── ResNet.py
│   │   └── ...
│   ├── config.py              # Configuration settings
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── model_utils.py         # Model training utilities
│   └── visualization.py       # Results plotting
├── run_count_prime.py         # Execution script for prime counting
└── run_cross_validation.py    # Execution script for cross-validation
```

## Advanced Usage

### Environment Setup with Flexible Imports

```python
# Flexible import approach - import only what you need
from smartpool import (
    limit_num_single_thread,    # Must be called first
    DataSize,                   # Available immediately
    move_optimizer_to,          # For optimizer management
    best_device                 # For dynamic device selection
)
limit_num_single_thread()  # Critical: prevent thread oversubscription

# Now safe to import scientific libraries
import torch
import numpy as np
```

### PyTorch Tensor Sharing

```python
# Set environment before imports
from smartpool import limit_num_single_thread
limit_num_single_thread()

from smartpool import ProcessPool
import torch

# Enable tensor sharing to avoid serialization overhead
with ProcessPool(use_torch=True) as pool:
    model = torch.nn.Linear(784, 10)
    data = torch.randn(1000, 784)
    
    # Tensors are shared directly between processes
    future = pool.submit(train_model, args=(model, data))
```

### Training Hot Migration

```python
# Environment setup
from smartpool import limit_num_single_thread, best_device
limit_num_single_thread()

import torch

def adaptive_training(model, data):
    # Automatically moves to best available device
    device = best_device()
    model = model.to(device)  # Direct model movement
    data = data.to(device)
    return train_epoch(model, data)

# Task automatically migrates from CPU to GPU when available
with ProcessPool() as pool:
    future = pool.submit(adaptive_training, args=(model, data))
```

### Custom Resource Requirements

```python
# Environment setup
from smartpool import limit_num_single_thread
limit_num_single_thread()

from smartpool import ProcessPool, DataSize

def train_model_fold(fold_data):
    # Your training code here
    return validation_score

# Specify resource requirements for ML training
with ProcessPool(max_workers=4, use_torch=True) as pool:
    future = pool.submit(
        train_model_fold,
        args=(fold_data,),
        need_cpu_cores=2,
        need_cpu_mem=2*DataSize.GB,
        need_gpu_cores=2048,    # Request 2048 CUDA cores (actual count)
        need_gpu_mem=4*DataSize.GB
    )
```

### Optimizer Management Best Practices

```python
# Complete workflow with proper device management
from smartpool import (
    limit_num_single_thread,
    move_optimizer_to,
    best_device
)
limit_num_single_thread()

import torch

# Initial setup
model = torch.nn.Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters())

# During training loop - handle device migrations
for epoch in range(epochs):
    # Check if device has changed (hot migration scenario)
    current_device = next(model.parameters()).device
    target_device = best_device()
    
    if current_device != target_device:
        model = model.to(target_device)
        optimizer = move_optimizer_to(optimizer, target_device)  # Critical!
    
    # Continue training...
```


## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure smartpool is installed: `pip install smartpool`
2. **GPU Not Detected**: Install pynvml: `pip install pynvml`
3. **Memory Errors**: Reduce batch size or number of workers
4. **Slow Performance**: Check resource requirements match actual usage
5. **Tensor Sharing Issues**: Ensure `use_torch=True` is set for PyTorch operations
6. **Thread Oversubscription**: Call `limit_num_single_thread()` before importing torch/numpy
7. **Optimizer State Loss**: Always use `move_optimizer_to()` during device migrations

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)


# Verify device selection
from smartpool import best_device
print(f"Best device: {best_device()}")

# Check environment variables are set
import os
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
```

## Contributing

Feel free to submit pull requests with:
- New example use cases
- Performance improvements
- Additional model architectures
- Better visualization tools

## License

MIT License - see main smartpool repository for details