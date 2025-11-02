"""
Quantum Anharmonic Oscillator Example
Based on the PMM paper (arXiv:2401.11694v1)

Problem: Solve the quantum anharmonic oscillator Hamiltonian
H(g) = a†a + g(a† + a)⁴

where g is the coupling parameter and we want to find the eigenvalues (energy levels).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from parametric_matrix_models import LinearPMM


def create_ladder_operators(n_levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create creation and annihilation operators for a quantum harmonic oscillator.

    Args:
        n_levels: Number of energy levels (matrix dimension)

    Returns:
        a: Annihilation operator
        a_dag: Creation operator
    """
    # Annihilation operator: a|n> = sqrt(n)|n-1>
    a = np.zeros((n_levels, n_levels))
    for n in range(1, n_levels):
        a[n-1, n] = np.sqrt(n)

    # Creation operator: a†|n> = sqrt(n+1)|n+1>
    a_dag = a.T

    return a, a_dag


def compute_exact_hamiltonian(g: float, n_levels: int = 20) -> np.ndarray:
    """
    Compute the exact Hamiltonian H(g) = a†a + g(a† + a)⁴.

    Args:
        g: Coupling parameter
        n_levels: Number of energy levels to use

    Returns:
        Hamiltonian matrix
    """
    a, a_dag = create_ladder_operators(n_levels)

    # H0 = a†a (number operator)
    H0 = a_dag @ a

    # Interaction term: (a† + a)⁴
    x = a_dag + a
    x4 = x @ x @ x @ x

    # Full Hamiltonian
    H = H0 + g * x4

    return H


def generate_training_data(g_values: np.ndarray,
                          n_levels: int = 20,
                          num_eigenvalues: int = 2) -> np.ndarray:
    """
    Generate exact training data for the anharmonic oscillator.

    Args:
        g_values: Array of coupling parameters
        n_levels: Number of levels for exact calculation
        num_eigenvalues: Number of lowest eigenvalues to compute

    Returns:
        Array of eigenvalues (shape: len(g_values), num_eigenvalues)
    """
    eigenvalues_list = []

    for g in g_values:
        H = compute_exact_hamiltonian(g, n_levels)
        # Compute only the lowest eigenvalues for efficiency
        eigs = eigh(H, eigvals_only=True, subset_by_index=[0, num_eigenvalues-1])
        eigenvalues_list.append(eigs)

    return np.array(eigenvalues_list)


def main():
    """Run the quantum anharmonic oscillator example."""
    print("=" * 70)
    print("Quantum Anharmonic Oscillator with Parametric Matrix Models")
    print("=" * 70)

    # Parameters from the paper
    matrix_dim = 5  # 5×5 PMM
    g_min, g_max = -0.01, 0.01
    n_train = 10
    num_eigenvalues = 2  # Fit two lowest eigenvalues

    # Generate training data
    print(f"\n1. Generating training data...")
    print(f"   - Coupling range: [{g_min}, {g_max}]")
    print(f"   - Number of training points: {n_train}")
    print(f"   - Number of eigenvalues to fit: {num_eigenvalues}")

    g_train = np.linspace(g_min, g_max, n_train)
    y_train = generate_training_data(g_train, num_eigenvalues=num_eigenvalues)

    print(f"   - Training data shape: {y_train.shape}")

    # Create and train PMM
    print(f"\n2. Training Parametric Matrix Model...")
    print(f"   - Matrix dimension: {matrix_dim}×{matrix_dim}")

    pmm = LinearPMM(matrix_dim=matrix_dim, hermitian=True)
    result = pmm.fit(
        g_train.reshape(-1, 1),
        y_train,
        num_outputs=num_eigenvalues,
        loss='mse',
        method='L-BFGS-B',
        max_iter=5000,
        verbose=False
    )

    print(f"   - Training completed: {result['success']}")
    print(f"   - Final loss: {result['loss']:.6e}")
    print(f"   - Iterations: {result['iterations']}")

    # Test on fine grid including extrapolation
    print(f"\n3. Testing and extrapolation...")
    g_test = np.linspace(g_min * 1.5, g_max * 1.5, 100)
    y_test_exact = generate_training_data(g_test, num_eigenvalues=num_eigenvalues)
    y_test_pmm = pmm.predict(g_test.reshape(-1, 1), num_outputs=num_eigenvalues)

    # Compute errors
    mse = np.mean((y_test_exact - y_test_pmm) ** 2)
    mae = np.mean(np.abs(y_test_exact - y_test_pmm))
    print(f"   - Test MSE: {mse:.6e}")
    print(f"   - Test MAE: {mae:.6e}")

    # Plot results
    print(f"\n4. Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(num_eigenvalues):
        # Plot eigenvalue i
        ax = axes[i]
        ax.plot(g_test, y_test_exact[:, i], 'b-', linewidth=2, label='Exact', alpha=0.7)
        ax.plot(g_test, y_test_pmm[:, i], 'r--', linewidth=2, label='PMM', alpha=0.7)
        ax.scatter(g_train, y_train[:, i], c='black', s=50, zorder=5, label='Training data')
        ax.axvline(g_min, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(g_max, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Coupling parameter g', fontsize=12)
        ax.set_ylabel(f'E_{i} (Energy level {i})', fontsize=12)
        ax.set_title(f'Eigenvalue {i}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('quantum_oscillator_pmm.png', dpi=150, bbox_inches='tight')
    print(f"   - Plot saved to: quantum_oscillator_pmm.png")

    # Error analysis
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(num_eigenvalues):
        errors = np.abs(y_test_exact[:, i] - y_test_pmm[:, i])
        ax.semilogy(g_test, errors, linewidth=2, label=f'Eigenvalue {i}')

    ax.axvline(g_min, color='gray', linestyle=':', alpha=0.5, label='Training range')
    ax.axvline(g_max, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coupling parameter g', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('PMM Prediction Error', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quantum_oscillator_error.png', dpi=150, bbox_inches='tight')
    print(f"   - Error plot saved to: quantum_oscillator_error.png")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    from typing import Tuple
    main()
