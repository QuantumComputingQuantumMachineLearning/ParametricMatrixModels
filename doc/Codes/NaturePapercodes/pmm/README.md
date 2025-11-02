# Parametric Matrix Models (PMMs)

Implementation of Parametric Matrix Models based on the paper:
**"Parametric Matrix Models"** by Patrick Cook, Danny Jammooa, Morten Hjorth-Jensen, Daniel D. Lee, and Dean Lee
(arXiv:2401.11694v1)

## Overview

Parametric Matrix Models (PMMs) are a machine learning approach based on matrix equations that can learn complex systems using minimal training data. PMMs represent functions as eigenvalues of parameterized matrices:

```
M(c) = M0 + Σ c_i * M_i
```

where `c` is a vector of input parameters and `M0, M_i` are learned matrices.

## Installation

```bash
pip install -r requirements.txt
```

## Files

- `parametric_matrix_models.py` - Core PMM implementation
  - `ParametricMatrixModel` - Base class for general PMMs
  - `LinearPMM` - Simplified single-parameter PMM

- `example_quantum_oscillator.py` - Quantum anharmonic oscillator example
  - Demonstrates PMM on quantum physics problem
  - Uses 5×5 matrix with linear parametrization

- `example_mnist_clustering.py` - MNIST clustering with tensor network PMM
  - Unsupervised clustering of MNIST digits
  - Uses tensor network decomposition for efficiency

## Usage

### Basic PMM Example

```python
from parametric_matrix_models import LinearPMM
import numpy as np

# Create a PMM with 5×5 matrices
pmm = LinearPMM(matrix_dim=5)

# Generate training data
c_train = np.linspace(-1, 1, 10).reshape(-1, 1)
y_train = np.sin(c_train) * np.arange(1, 3)  # 2 outputs

# Train the model
pmm.fit(c_train, y_train, num_outputs=2, max_iter=1000)

# Make predictions
c_test = np.array([[0.5]])
predictions = pmm.predict(c_test, num_outputs=2)
```

### Running Examples

#### Quantum Anharmonic Oscillator

```bash
python example_quantum_oscillator.py
```

This will:
- Generate training data for H(g) = a†a + g(a† + a)⁴
- Train a 5×5 PMM on 10 data points
- Test extrapolation beyond training range
- Generate plots showing fit quality

Output files:
- `quantum_oscillator_pmm.png` - Eigenvalue predictions
- `quantum_oscillator_error.png` - Prediction errors

#### MNIST Clustering

```bash
python example_mnist_clustering.py
```

This will:
- Load MNIST dataset (1000 samples by default)
- Train tensor network PMM for dimensionality reduction
- Generate 2D embedding visualization
- Save clustering plot

Output files:
- `mnist_clustering_pmm.png` - 2D clustering visualization

## Key Features

1. **Minimal Training Data**: PMMs can learn from small datasets
2. **Extrapolation**: Can predict beyond training range
3. **Universal Approximation**: Theoretically can approximate any function
4. **Interpretability**: Matrix structure provides mathematical insight
5. **Flexibility**: Applicable to scientific computing and ML problems

## Paper Reference

```bibtex
@article{cook2024parametric,
  title={Parametric Matrix Models},
  author={Cook, Patrick and Jammooa, Danny and Hjorth-Jensen, Morten and Lee, Daniel D. and Lee, Dean},
  journal={arXiv preprint arXiv:2401.11694},
  year={2024}
}
```

## License

MIT License - See paper for original research citation requirements.
