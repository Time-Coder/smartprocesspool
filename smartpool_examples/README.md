# SmartPool Examples

This package contains practical examples demonstrating the capabilities of SmartPool for various computational tasks.

## Examples Overview

### 1. Prime Number Counting (`count_prime`)

Count the number of prime numbers below 10000 using smartpool.ProcessPool.
Demonstrates basic usage of smartpool.ProcessPool.

#### Running the Example

```bash
python -m smartpool_examples.count_prime
```

### 2. Cross-Validation for Deep Learning models (`cross_validation`)

Demonstrates SmartPool's capabilities for machine learning workloads with GPU resource management.

#### Running the Example

```bash
# Using ProcessPool  
python -m smartpool_examples.cross_validation --pool smartpool.ProcessPool

# Using ThreadPool
python -m smartpool_examples.cross_validation --pool smartpool.ThreadPool

# Using multiprocessing.Pool
python -m smartpool_examples.cross_validation --pool multiprocessing.Pool

# Using concurrent.futures.ProcessPoolExecutor
python -m smartpool_examples.cross_validation --pool concurrent.futures.ProcessPoolExecutor

# Using concurrent.futures.ThreadPoolExecutor
python -m smartpool_examples.cross_validation --pool concurrent.futures.ThreadPoolExecutor

# Using joblib.Parallel(backend='loky')
python -m smartpool_examples.cross_validation --pool joblib.Parallel(backend='loky')

# Using joblib.Parallel(backend='threading')
python -m smartpool_examples.cross_validation --pool joblib.Parallel(backend='threading')

# Using Ray
python -m smartpool_examples.cross_validation --pool ray
```

#### What it Demonstrates

- GPU memory management and core allocation
- Automatic device selection (CPU vs GPU)
- Cross-validation pipeline parallelization
- Resource monitoring during training
- Performance comparison with external frameworks


## License

MIT License - see main smartpool repository for details