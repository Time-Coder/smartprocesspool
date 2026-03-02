# SmartPool

SmartPool is a Python library that provides intelligent resource-aware pooling mechanisms for parallel and distributed computing. It automatically manages CPU and GPU resources to optimize performance while preventing resource exhaustion.

## Features

- **Multiple Pool Types**: ProcessPool, ThreadPool, and InterpreterPool for different use cases
- **Automatic Resource Management**: Monitors and manages CPU cores, memory, and GPU resources
- **Hardware-Aware Scheduling**: Automatically detects system resources and schedules tasks accordingly
- **PyTorch Integration**: Built-in support for PyTorch multiprocessing with tensor sharing to avoid serialization
- **Training Hot Migration**: Automatically moves CPU training tasks to GPU when best_device() changes
- **Cross-Platform Compatibility**: Works on Windows, Linux, and macOS

## Installation

```bash
pip install smartpool
```

## Quick Start

### Basic Usage

```python
from smartpool import ProcessPool

# Create a process pool that automatically manages system resources
with ProcessPool(max_workers=4) as pool:
    # Submit tasks with proper argument passing
    futures = [pool.submit(expensive_computation, args=(arg,)) for arg in arguments]
    
    # Get results
    results = [future.result() for future in futures]
```

### Resource-Aware Task Scheduling

```python
from smartpool import ProcessPool, DataSize

# Tasks can specify their resource requirements
def memory_intensive_task(data):
    # Your computation here
    return processed_data

with ProcessPool(max_workers=8) as pool:
    # Pool automatically schedules tasks based on available memory
    future = pool.submit(
        memory_intensive_task, 
        args=(large_dataset,),
        need_cpu_cores=2,           # Request 2 CPU cores
        need_cpu_mem=4*DataSize.GB, # Request 4GB RAM
        need_gpu_cores=1024,        # Request 1024 CUDA cores (NOT percentage)
        need_gpu_mem=2*DataSize.GB  # Request 2GB GPU memory
    )
```

## Core Components

### ProcessPool
Optimized for CPU-intensive tasks with automatic resource monitoring:

```python
from smartpool import ProcessPool

pool = ProcessPool(
    max_workers=0,              # 0 means auto-detect based on system resources
    process_name_prefix="MyApp:",
    mp_context="spawn",         # Multiprocessing context
    use_torch=True             # Enable PyTorch multiprocessing with tensor sharing
)
```

### ThreadPool  
Lightweight option for I/O-bound tasks:

```python
from smartpool import ThreadPool

pool = ThreadPool(
    max_workers=16,
    thread_name_prefix="IOThread:"
)
```

### InterpreterPool
Uses Python's subinterpreters for isolated execution (Python 3.14+):

```python
from smartpool import InterpreterPool

pool = InterpreterPool(max_workers=4)
```

## Advanced Features

### PyTorch Tensor Sharing

When `use_torch=True`, SmartPool enables PyTorch's tensor sharing mechanism to avoid costly serialization:

```python
# Set environment before importing scientific libraries
from smartpool import limit_num_single_thread, ProcessPool
limit_num_single_thread()  # Set environment variables to prevent oversubscription

import torch
from smartpool import DataSize

# Tensors are shared directly between processes without serialization
model = torch.nn.Linear(100, 10)
data = torch.randn(32, 100)

with ProcessPool(use_torch=True) as pool:
    # Proper argument passing for submit
    future = pool.submit(train_step, args=(model, data))
```

### Training Hot Migration with Optimizer State Management

SmartPool automatically migrates training tasks from CPU to GPU when better devices become available, with complete optimizer state handling:

```python
# Complete setup for training with optimizer migration
from smartpool import (
    limit_num_single_thread, 
    best_device, 
    move_optimizer_to,
    ProcessPool
)
# Critical: Call before importing torch/numpy
limit_num_single_thread()

import torch

def training_task(model, optimizer, data):
    # Training automatically moves to best available device
    device = best_device()  # Dynamically detect optimal device
    
    # Move both model and optimizer to target device
    model = model.to(device)
    # Essential: Move optimizer state to maintain training continuity
    optimizer = move_optimizer_to(optimizer, device)
    
    data = data.to(device)
    # Training proceeds on optimal device
    return train_model(model, optimizer, data)

with ProcessPool() as pool:
    # Task starts on CPU, automatically migrates to GPU when available
    future = pool.submit(training_task, args=(model, optimizer, data))
```

### Utility Functions

```python
# Flexible import approach - call limit_num_single_thread first
from smartpool import (
    limit_num_single_thread,    # Environment control
    DataSize,                   # Memory constants
    move_optimizer_to,          # Optimizer device management
    best_device,                # Dynamic device selection
)
limit_num_single_thread()  # Prevent oversubscription

import torch

# Example usage
device = best_device()     # Get best available device (cuda/cpu)

# Critical: Move optimizer during device migration
optimizer = move_optimizer_to(optimizer, device)

# Model movement (standard PyTorch approach)
model = model.to(device)
```

## Configuration Options

### Pool Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_workers` | int | Maximum number of workers (0 = auto-detect) |
| `initializer` | callable | Function to initialize each worker |
| `initargs` | tuple | Arguments for initializer |
| `max_tasks_per_child` | int | Maximum tasks per worker before restart |
| `use_torch` | bool | Enable PyTorch multiprocessing with tensor sharing |

### Task Resource Requirements

| Parameter | Type | Description |
|-----------|------|-------------|
| `need_cpu_cores` | int | CPU cores required |
| `need_cpu_mem` | int | Memory required (bytes) |
| `need_gpu_cores` | int | **CUDA cores required (NOT percentage)** |
| `need_gpu_mem` | int | GPU memory required (bytes) |

## Correct Submit Method Usage

**Important**: SmartPool's submit method differs from concurrent.futures:

```python
# Incorrect (concurrent.futures style)
future = pool.submit(func, arg1, arg2, arg3)

# Correct (SmartPool style)
future = pool.submit(func, args=(arg1, arg2, arg3))

# For single argument
future = pool.submit(func, args=(single_arg,))

# For no arguments
future = pool.submit(func, args=())
```

## Performance Tips

1. **Environment Setup**: Call `limit_num_single_thread()` before importing torch/numpy
2. **Optimizer Management**: Use `move_optimizer_to()` during device migrations to preserve training state
3. **Enable PyTorch Integration**: Use `use_torch=True` for tensor sharing
4. **Dynamic Device Selection**: Use `best_device()` for automatic migration
5. **Resource Estimation**: Provide accurate resource requirements for better scheduling
6. **Batch Processing**: Group similar tasks together for efficiency

## License

MIT License