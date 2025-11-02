"""
Parametric Matrix Models (PMMs)
Implementation based on "Parametric Matrix Models" (arXiv:2401.11694v1)
Authors: Patrick Cook, Danny Jammooa, Morten Hjorth-Jensen, Daniel D. Lee, Dean Lee
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from typing import List, Tuple, Callable, Optional
import warnings


class ParametricMatrixModel:
    """
    Base class for Parametric Matrix Models.

    PMMs represent functions as eigenvalues of parameterized matrices:
    M(c) = M0 + sum_i(c_i * M_i)

    where c is a vector of input parameters and M0, M_i are learned matrices.
    """

    def __init__(self, matrix_dim: int, num_params: int, hermitian: bool = True):
        """
        Initialize a Parametric Matrix Model.

        Args:
            matrix_dim: Dimension of the square matrix
            num_params: Number of input parameters
            hermitian: Whether to enforce Hermitian (symmetric) matrices
        """
        self.matrix_dim = matrix_dim
        self.num_params = num_params
        self.hermitian = hermitian

        # Initialize M0 and M_i matrices randomly
        self.M0 = self._init_matrix()
        self.M_params = [self._init_matrix() for _ in range(num_params)]

    def _init_matrix(self) -> np.ndarray:
        """Initialize a random matrix (Hermitian if specified)."""
        M = np.random.randn(self.matrix_dim, self.matrix_dim) * 0.01
        if self.hermitian:
            M = (M + M.T) / 2  # Make symmetric/Hermitian
        return M

    def get_matrix(self, c: np.ndarray) -> np.ndarray:
        """
        Compute M(c) = M0 + sum_i(c_i * M_i).

        Args:
            c: Input parameters (shape: num_params,)

        Returns:
            Parameterized matrix M(c)
        """
        M = self.M0.copy()
        for i, c_i in enumerate(c):
            M += c_i * self.M_params[i]
        return M

    def get_eigenvalues(self, c: np.ndarray, sort: bool = True) -> np.ndarray:
        """
        Compute eigenvalues of M(c).

        Args:
            c: Input parameters
            sort: Whether to sort eigenvalues in ascending order

        Returns:
            Eigenvalues of M(c)
        """
        M = self.get_matrix(c)
        if self.hermitian:
            eigenvalues = eigh(M, eigvals_only=True)
        else:
            eigenvalues = np.linalg.eigvals(M)

        if sort:
            eigenvalues = np.sort(eigenvalues)
        return eigenvalues

    def predict(self, c: np.ndarray, num_outputs: Optional[int] = None) -> np.ndarray:
        """
        Predict outputs for given input parameters.
        Uses the lowest (or specified number of) eigenvalues as outputs.

        Args:
            c: Input parameters (can be 1D or 2D for batch)
            num_outputs: Number of eigenvalues to return (default: all)

        Returns:
            Eigenvalues (outputs)
        """
        c = np.atleast_2d(c)
        if c.shape[1] != self.num_params and c.shape[0] == self.num_params:
            c = c.T

        results = []
        for c_i in c:
            eigs = self.get_eigenvalues(c_i)
            if num_outputs is not None:
                eigs = eigs[:num_outputs]
            results.append(eigs)

        return np.array(results)

    def _flatten_matrices(self) -> np.ndarray:
        """Flatten all matrix parameters into a single vector."""
        params = [self.M0.flatten()]
        for M_i in self.M_params:
            params.append(M_i.flatten())
        return np.concatenate(params)

    def _unflatten_matrices(self, flat_params: np.ndarray):
        """Unflatten parameter vector back into matrices."""
        n_elements = self.matrix_dim ** 2
        offset = 0

        self.M0 = flat_params[offset:offset+n_elements].reshape(self.matrix_dim, self.matrix_dim)
        if self.hermitian:
            self.M0 = (self.M0 + self.M0.T) / 2
        offset += n_elements

        for i in range(self.num_params):
            M_i = flat_params[offset:offset+n_elements].reshape(self.matrix_dim, self.matrix_dim)
            if self.hermitian:
                M_i = (M_i + M_i.T) / 2
            self.M_params[i] = M_i
            offset += n_elements

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            num_outputs: Optional[int] = None,
            loss: str = 'mse',
            method: str = 'L-BFGS-B',
            max_iter: int = 1000,
            verbose: bool = False) -> dict:
        """
        Train the PMM to fit data.

        Args:
            X: Input parameters (shape: n_samples, num_params)
            y: Target outputs (shape: n_samples, num_outputs)
            num_outputs: Number of eigenvalues to use (default: inferred from y)
            loss: Loss function ('mse' or 'kl')
            method: Optimization method
            max_iter: Maximum iterations
            verbose: Whether to print progress

        Returns:
            Optimization result dictionary
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)

        if num_outputs is None:
            num_outputs = y.shape[1] if len(y.shape) > 1 else 1

        def loss_function(flat_params):
            """Compute loss for current parameters."""
            self._unflatten_matrices(flat_params)

            y_pred = self.predict(X, num_outputs)

            if loss == 'mse':
                return np.mean((y - y_pred) ** 2)
            elif loss == 'kl':
                # Kullback-Leibler divergence (for probability distributions)
                y_norm = y / (np.sum(y, axis=1, keepdims=True) + 1e-10)
                y_pred_norm = y_pred / (np.sum(y_pred, axis=1, keepdims=True) + 1e-10)
                return np.sum(y_norm * np.log((y_norm + 1e-10) / (y_pred_norm + 1e-10)))
            else:
                raise ValueError(f"Unknown loss function: {loss}")

        # Initial parameters
        x0 = self._flatten_matrices()

        # Optimize
        result = minimize(
            loss_function,
            x0,
            method=method,
            options={'maxiter': max_iter, 'disp': verbose}
        )

        # Set optimized parameters
        self._unflatten_matrices(result.x)

        return {
            'success': result.success,
            'loss': result.fun,
            'iterations': result.nit,
            'message': result.message
        }


class LinearPMM(ParametricMatrixModel):
    """
    Simplified PMM for single parameter: M(c) = M0 + c * M1
    Useful for problems with one input dimension.
    """

    def __init__(self, matrix_dim: int, hermitian: bool = True):
        super().__init__(matrix_dim, num_params=1, hermitian=hermitian)

    def get_matrix(self, c: float) -> np.ndarray:
        """Compute M(c) = M0 + c * M1."""
        if isinstance(c, np.ndarray):
            c = c[0] if c.size == 1 else c
        return self.M0 + c * self.M_params[0]

    def get_eigenvalues(self, c: float, sort: bool = True) -> np.ndarray:
        """Compute eigenvalues of M(c)."""
        M = self.get_matrix(c)
        if self.hermitian:
            eigenvalues = eigh(M, eigvals_only=True)
        else:
            eigenvalues = np.linalg.eigvals(M)

        if sort:
            eigenvalues = np.sort(eigenvalues)
        return eigenvalues


if __name__ == "__main__":
    # Simple test
    print("Testing Parametric Matrix Model...")

    # Create a simple PMM
    pmm = LinearPMM(matrix_dim=5)

    # Test prediction
    c_test = np.array([0.5])
    eigs = pmm.get_eigenvalues(c_test)
    print(f"Eigenvalues at c={c_test[0]}: {eigs}")

    # Create synthetic training data
    c_train = np.linspace(-1, 1, 10).reshape(-1, 1)
    y_train = np.array([np.sin(c) * np.arange(1, 3) for c in c_train.flatten()])

    # Train the model
    print("\nTraining PMM...")
    result = pmm.fit(c_train, y_train, num_outputs=2, verbose=True)
    print(f"Training result: {result}")

    # Test prediction after training
    c_test = np.array([[0.0], [0.5]])
    predictions = pmm.predict(c_test, num_outputs=2)
    print(f"\nPredictions at c={c_test.flatten()}: {predictions}")
