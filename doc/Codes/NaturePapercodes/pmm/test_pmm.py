"""
Simple test script to verify PMM implementation.
"""

import numpy as np
from parametric_matrix_models import ParametricMatrixModel, LinearPMM


def test_basic_pmm():
    """Test basic PMM functionality."""
    print("Testing basic PMM...")

    # Create PMM
    pmm = ParametricMatrixModel(matrix_dim=4, num_params=2)

    # Test matrix computation
    c = np.array([0.5, -0.3])
    M = pmm.get_matrix(c)
    assert M.shape == (4, 4), "Matrix shape incorrect"

    # Test eigenvalue computation
    eigs = pmm.get_eigenvalues(c)
    assert len(eigs) == 4, "Wrong number of eigenvalues"

    print("  ✓ Basic PMM test passed")


def test_linear_pmm():
    """Test LinearPMM functionality."""
    print("Testing LinearPMM...")

    # Create LinearPMM
    pmm = LinearPMM(matrix_dim=5)

    # Test prediction
    c = np.array([0.0])
    eigs = pmm.get_eigenvalues(c)
    assert len(eigs) == 5, "Wrong number of eigenvalues"

    print("  ✓ LinearPMM test passed")


def test_training():
    """Test training functionality."""
    print("Testing training...")

    # Create synthetic data
    np.random.seed(42)
    c_train = np.linspace(-1, 1, 20).reshape(-1, 1)
    y_train = np.column_stack([
        c_train.flatten()**2 + 0.5,
        c_train.flatten()**2 + 1.5
    ])

    # Train PMM
    pmm = LinearPMM(matrix_dim=6)
    result = pmm.fit(c_train, y_train, num_outputs=2, max_iter=1000)

    assert result['success'], "Training failed"
    print(f"  Final loss: {result['loss']:.6e}")

    # Test prediction
    c_test = np.array([[0.0]])
    pred = pmm.predict(c_test, num_outputs=2)
    expected = np.array([[0.5, 1.5]])

    error = np.abs(pred - expected).max()
    print(f"  Prediction error: {error:.6e}")

    # Allow reasonable tolerance - PMMs may not fit perfectly but should be reasonable
    assert error < 0.5, f"Prediction error too large: {error}"

    print("  ✓ Training test passed")


def test_hermitian():
    """Test Hermitian matrix property."""
    print("Testing Hermitian property...")

    pmm = LinearPMM(matrix_dim=4, hermitian=True)
    c = np.array([0.7])
    M = pmm.get_matrix(c)

    # Check if matrix is symmetric
    is_symmetric = np.allclose(M, M.T)
    assert is_symmetric, "Matrix is not Hermitian"

    print("  ✓ Hermitian test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running PMM Tests")
    print("=" * 60)
    print()

    try:
        test_basic_pmm()
        test_linear_pmm()
        test_hermitian()
        test_training()

        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
