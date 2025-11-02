import numpy as np
from numpy.linalg import eigh

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def random_hermitian(n, rng):
    """
    Generate a random n x n Hermitian matrix with complex entries.
    H = A + A^† ensures Hermiticity.
    """
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2.0
    return H.real if np.allclose(H.imag, 0.0) else H  # keep Hermitian (may be complex Hermitian)


class ParametricMatrixModel:
    """
    Minimal affine parametric matrix model:
        M(theta) = M0 + theta * M1
    inspired by the affine PMM form M({c_l}) = M0 + sum_l c_l M_l
    discussed in the parametric matrix models paper.   [oai_citation:1‡arXiv](https://arxiv.org/pdf/2401.11694)

    Here:
    - n_dim: dimension of the full Hermitian matrix (e.g. 100)
    - We use a single scalar parameter 'theta' for simplicity.
    """

    def __init__(self, n_dim=100, seed=0):
        self.n_dim = n_dim
        rng = np.random.default_rng(seed)

        # Trainable / learnable matrices in the PMM language:
        # Here we just generate them randomly.
        self.M0 = random_hermitian(n_dim, rng)
        self.M1 = random_hermitian(n_dim, rng)

    def matrix(self, theta):
        """
        Return the full n_dim x n_dim Hermitian matrix M(theta).
        """
        return self.M0 + theta * self.M1


def lowest_eigensystem_full(H, r=5):
    """
    Solve the full eigenvalue problem H v = E v for a Hermitian H,
    return the r lowest eigenvalues and eigenvectors.

    We use numpy.linalg.eigh, which returns all eigenpairs sorted ascending.
    For n=100 this is cheap and numerically stable.
    """
    evals, evecs = eigh(H)
    return evals[:r], evecs[:, :r]  # shapes: (r,), (n,r)


def build_reduced_basis(model, theta_ref, r=5):
    """
    Build the reduced subspace ("model order reduction"):
    1. Compute the r lowest eigenvectors of M(theta_ref)
    2. Orthonormalize them (eigh already gives orthonormal vecs)
    3. Return the basis matrix V of shape (n, r)

    In PMM language, this is selecting the low-lying eigenvectors
    of the primary matrix as the relevant subspace for predictions.  [oai_citation:2‡arXiv](https://arxiv.org/pdf/2401.11694)
    """
    H_ref = model.matrix(theta_ref)
    _, V = lowest_eigensystem_full(H_ref, r=r)  # V is n x r, columns orthonormal
    return V


def project_effective_matrix(model, theta, V):
    """
    Given:
      - model: provides full M(theta) (n x n Hermitian)
      - V: reduced basis (n x r) whose columns are orthonormal
    Return:
      Heff(theta) = V^† M(theta) V, which is r x r Hermitian.

    This is the effective Hamiltonian / reduced matrix.
    """
    H_full = model.matrix(theta)             # (n,n)
    Heff = V.conj().T @ H_full @ V           # (r,r)
    # Ensure Hermiticity numerically
    Heff = (Heff + Heff.conj().T) / 2.0
    return Heff


def lowest_eigensystem_reduced(Heff):
    """
    Solve the reduced r x r eigenvalue problem.
    Returns eigenvalues (ascending) and eigenvectors in the reduced space.
    """
    evals_eff, evecs_eff = eigh(Heff)
    return evals_eff, evecs_eff  # shapes: (r,), (r,r)


# ------------------------------------------------------------
# Demonstration / example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. Build a 100x100 parametric Hermitian model
    model = ParametricMatrixModel(n_dim=100, seed=42)

    # 2. Pick a reference parameter theta_ref where we'll build the reduced basis
    theta_ref = 0.0

    # 3. Build reduced basis (keep five lowest modes)
    r = 5
    V = build_reduced_basis(model, theta_ref, r=r)  # shape (100,5)

    # 4. Now evaluate spectra for various theta using:
    #    (a) exact low-lying eigenvalues from full 100x100 matrix
    #    (b) approximate low-lying eigenvalues from 5x5 effective matrix
    thetas = np.linspace(-1.0, 1.0, 5)

    print("theta |  full eigenvalues (lowest 5)         |  reduced (5x5) eigenvalues")
    print("------|--------------------------------------|--------------------------------")
    for theta in thetas:
        H_full = model.matrix(theta)
        evals_full, _ = lowest_eigensystem_full(H_full, r=r)

        Heff = project_effective_matrix(model, theta, V)
        evals_eff, _ = lowest_eigensystem_reduced(Heff)

        # sort both (eigh already sorted ascending, but after projection
        # there could be slight numerical differences, so we'll sort anyway)
        evals_full_sorted = np.sort(evals_full)
        evals_eff_sorted = np.sort(evals_eff)

        print(f"{theta:5.2f} | {evals_full_sorted} | {evals_eff_sorted}")

    # Notes:
    # - At theta_ref, the reduced model is exact for those 5 modes,
    #   because V was built from those exact eigenvectors.
    # - Away from theta_ref, the 5x5 effective matrix gives an approximation
    #   to the lowest 5 eigenvalues without diagonalizing the full 100x100.
