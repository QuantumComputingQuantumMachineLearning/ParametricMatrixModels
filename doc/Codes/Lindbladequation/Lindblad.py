"""
Lindblad / open-quantum-systems examples (+ validation + analytic thermal state)

Implements:
  • 1-qubit & 2-qubit mixed states with expectation values
  • partial traces
  • Liouvillian construction (vectorized, superoperator matrix)
  • steady-state solver  L|ρ⟩⟩ = 0 with Tr(ρ)=1
  • harmonic oscillator coupled to a bath (two Lindblad jumps)
    – Liouvillian spectrum
    – time evolution of ⟨n⟩(t) from a Fock state
    – analytic thermal steady state (mean occupancy τ) and comparison to numerical ρ_ss

"""

from __future__ import annotations
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ---------- Basic linear-algebra helpers ----------
def dag(A: np.ndarray) -> np.ndarray:
    return A.conj().T

def kron(*ops: np.ndarray) -> np.ndarray:
    M = np.array([[1.0+0j]])
    for op in ops:
        M = np.kron(M, op)
    return M

def vec(A: np.ndarray) -> np.ndarray:
    return A.reshape((-1, 1), order="F")

def unvec(v: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return v.reshape(shape, order="F")

def projector(ket: np.ndarray) -> np.ndarray:
    return ket @ ket.conj().T

# ---------- Qubit basis & Pauli ----------
zero = np.array([[1.0+0j],[0.0+0j]])
one  = np.array([[0.0+0j],[1.0+0j]])
plus = (zero + one)/np.sqrt(2.0)
minus= (zero - one)/np.sqrt(2.0)

I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

# ---------- Example 3.1: One-qubit mixed state ----------
def one_qubit_mixed_density() -> np.ndarray:
    # 50% |0⟩ + 25% |+⟩ + 25% |−⟩  →  ρ = diag(3/4, 1/4)
    rho = 0.5*projector(zero) + 0.25*projector(plus) + 0.25*projector(minus)
    return rho

# ---------- Example 3.2: Two-qubit mixed state ----------
def bell_pm(sign: int = +1) -> np.ndarray:
    # |B±> = (|00> ± |11>)/√2
    return (kron(zero, zero) + sign*kron(one, one))/np.sqrt(2.0)

def two_qubit_mixed_density() -> np.ndarray:
    """
    ρ = 0.9 |B+⟩⟨B+| + 0.1 * [ (3/6)|01⟩⟨01| + (2/6)|10⟩⟨10| + (1/6)|B-⟩⟨B-| ].
    """
    Bp = bell_pm(+1); Bm = bell_pm(-1)
    psi01 = kron(zero, one); psi10 = kron(one, zero)
    rho = (0.9 * projector(Bp)
           + 0.1*(3/6)*projector(psi01)
           + 0.1*(2/6)*projector(psi10)
           + 0.1*(1/6)*projector(Bm))
    return rho

def expect(op: np.ndarray, rho: np.ndarray) -> complex:
    return np.trace(rho @ op)

def two_qubit_expectations(rho: np.ndarray) -> Dict[str, float]:
    vals = {
        "<σx1>": (expect(kron(sx, I2), rho)).real,
        "<σy1>": (expect(kron(sy, I2), rho)).real,
        "<σz1>": (expect(kron(sz, I2), rho)).real,
        "<σx2>": (expect(kron(I2, sx), rho)).real,
        "<σy2>": (expect(kron(I2, sy), rho)).real,
        "<σz2>": (expect(kron(I2, sz), rho)).real,
        "<σx1σx2>": (expect(kron(sx, sx), rho)).real,
        "<σy1σy2>": (expect(kron(sy, sy), rho)).real,
        "<σz1σz2>": (expect(kron(sz, sz), rho)).real,
    }
    return vals

def two_qubit_expected_targets() -> Dict[str, float]:
    """
    Analytic targets for the mixture defined above:
      <σx1>=<σy1>=<σx2>=<σy2>=0
      <σz1>=+1/60, <σz2>=-1/60
      <σx1σx2>=+53/60, <σy1σy2>=-53/60, <σz1σz2>=5/6
    """
    return {
        "<σx1>": 0.0,
        "<σy1>": 0.0,
        "<σx2>": 0.0,
        "<σy2>": 0.0,
        "<σz1>": 1.0/60.0,
        "<σz2>": -1.0/60.0,
        "<σx1σx2>": 53.0/60.0,
        "<σy1σy2>": -53.0/60.0,
        "<σz1σz2>": 5.0/6.0,
    }

def validate_two_qubit_expectations(rho: np.ndarray, atol: float = 1e-10) -> Dict[str, Dict[str, float]]:
    """
    Returns a dict mapping observable -> {"numeric": ..., "target": ..., "error": ...}
    """
    num = two_qubit_expectations(rho)
    tgt = two_qubit_expected_targets()
    out = {}
    for k in tgt.keys():
        err = float(num[k] - tgt[k])
        out[k] = {"numeric": float(num[k]), "target": float(tgt[k]), "error": err}
    return out

# ---------- Partial trace (2-part system) ----------
def partial_trace(rho_AB: np.ndarray, dims: Tuple[int, int], trace_out: str = "B") -> np.ndarray:
    """
    Trace out subsystem 'trace_out' ∈ {'A','B'} from bipartite density matrix ρ_AB.
    dims=(dA, dB).
    """
    dA, dB = dims
    rho = rho_AB.reshape(dA, dB, dA, dB)  # (a, b, a', b')
    if trace_out.upper() == "B":
        return np.tensordot(rho, np.eye(dB), axes=([1,3],[0,1]))
    elif trace_out.upper() == "A":
        return np.tensordot(rho, np.eye(dA), axes=([0,2],[0,1]))
    else:
        raise ValueError("trace_out must be 'A' or 'B'")

# ---------- Liouvillian: Lindblad superoperator matrix ----------
def liouvillian(H: np.ndarray, jumps: List[np.ndarray], rates: List[float]) -> np.ndarray:
    """
    L = -i (I ⊗ H - H^T ⊗ I)
        + Σ_k γ_k [ (L_k^* ⊗ L_k) - ½ (I ⊗ L_k^†L_k) - ½ ( (L_k^†L_k)^T ⊗ I ) ].
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j*(np.kron(I, H) - np.kron(H.T, I))
    for Lk, g in zip(jumps, rates):
        LdagL = Lk.conj().T @ Lk
        L += g*(np.kron(Lk.conj(), Lk) - 0.5*np.kron(I, LdagL) - 0.5*np.kron(LdagL.T, I))
    return L

def evolve_liouvillian(L: np.ndarray, rho0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Evolve d|ρ⟩⟩/dt = L|ρ⟩⟩ using expm_multiply over a range of times.
    Returns array with shape (len(times), d, d).
    """
    d = rho0.shape[0]
    v0 = vec(rho0).ravel()
    evols = expm_multiply(L, v0, start=0.0, stop=float(times[-1]), num=len(times), endpoint=True)
    out = np.empty((len(times), d, d), dtype=complex)
    for i, v in enumerate(evols):
        out[i] = unvec(v.reshape((-1,1), order="F"), (d, d))
    return out

def steady_state(L: np.ndarray, dim: int) -> np.ndarray:
    """
    Solve for ρ_ss s.t. L|ρ⟩⟩ = 0 and Tr(ρ)=1 via least squares with trace constraint.
    """
    D = dim*dim
    A = np.zeros((D+1, D), dtype=complex)
    b = np.zeros((D+1,), dtype=complex)
    A[:D, :] = L
    tr_row = np.zeros((dim, dim), dtype=complex); np.fill_diagonal(tr_row, 1.0)
    A[D, :] = vec(tr_row).ravel()
    b[D] = 1.0
    x, *_ = la.lstsq(A, b, lapack_driver='gelsy')
    rho = unvec(x.reshape((-1,1)), (dim, dim))
    rho = 0.5*(rho + rho.conj().T)  # Hermitize
    w, V = la.eigh(rho)
    w = np.clip(w.real, 0.0, None); s = max(1e-15, w.sum()); w = w/s
    rho = V @ np.diag(w) @ V.conj().T
    return rho

# ---------- Harmonic oscillator operators ----------
def destroy(dim: int) -> np.ndarray:
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        a[n-1, n] = np.sqrt(n)
    return a

def create(dim: int) -> np.ndarray:
    return destroy(dim).conj().T

def number_op(dim: int) -> np.ndarray:
    n = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        n[k,k] = k
    return n

def fock(n: int, dim: int) -> np.ndarray:
    v = np.zeros((dim,1), dtype=complex); v[n,0] = 1.0
    return v

# ---------- Harmonic oscillator coupled to a thermal bath ----------
def ho_lindblad(omega: float = 1.0, gamma: float = 0.1, tau: float = 2.0, nmax: int = 20):
    """
    H = ω a†a
    L1 = √(τ+1) a,   L2 = √τ a†,  with rates γ1=γ2=γ
    """
    dim = nmax + 1
    a = destroy(dim); ad = a.conj().T
    H = omega * (ad @ a)
    L1 = np.sqrt(tau + 1.0) * a
    L2 = np.sqrt(tau) * ad
    jumps = [L1, L2]; rates = [gamma, gamma]
    return H, jumps, rates

def thermal_state(dim: int, tau: float) -> np.ndarray:
    """
    Analytic thermal (geometric) state with mean occupancy τ:
      p_n ∝ (τ/(1+τ))^n / (1+τ)
    Truncated to 'dim' and renormalized within the cutoff.
    """
    r = tau/(1.0 + tau) if tau >= 0 else 0.0
    p = np.array([(1.0/(1.0+tau)) * (r**n) for n in range(dim)], dtype=float)
    Z = p.sum()
    p = p / max(1e-15, Z)
    rho = np.diag(p.astype(complex))
    return rho

def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """½ ||ρ - σ||_1 via eigenvalues of |ρ-σ| (dense)."""
    delta = rho - sigma
    w = la.eigvals(delta)
    # For Hermitian delta, use singular values; enforce Hermitian
    deltaH = 0.5*(delta + delta.conj().T)
    s = la.svd(deltaH, compute_uv=False)
    return 0.5*float(np.sum(s.real))

def ho_example_run(omega=1.0, gamma=0.1, tau=2.0, nmax=12, n0=5,
                   t_final=20.0, num_t=120, do_plots=True, figure_prefix="ho"):
    """
    Compute Liouvillian spectrum and ⟨n⟩(t) for a truncated harmonic oscillator,
    plus analytic vs numerical steady state comparison.
    """
    H, jumps, rates = ho_lindblad(omega, gamma, tau, nmax)
    L = liouvillian(H, jumps, rates)

    # Liouvillian spectrum
    evals = la.eigvals(L)
    if do_plots:
        plt.figure()
        plt.scatter(evals.real, evals.imag, s=10)
        plt.xlabel("Re(λ)")
        plt.ylabel("Im(λ)")
        plt.title("Liouvillian Spectrum")
        plt.tight_layout()

    # Time evolution ⟨n⟩(t) from |n0⟩
    dim = nmax + 1
    n_op = number_op(dim)
    rho0 = projector(fock(n0, dim))
    ts = np.linspace(0.0, float(t_final), num_t)
    rhos = evolve_liouvillian(L, rho0, ts)
    n_t = np.real([np.trace(r @ n_op) for r in rhos])

    if do_plots:
        plt.figure()
        plt.plot(ts, n_t, lw=2)
        plt.xlabel("Time")
        plt.ylabel("<n>")
        plt.title("Time evolution of ⟨n⟩")
        plt.tight_layout()

    # Steady states: numerical and analytic (thermal)
    rho_ss_num = steady_state(L, dim)
    rho_ss_th  = thermal_state(dim, tau)

    n_num = float(np.real(np.trace(rho_ss_num @ n_op)))
    n_th  = float(np.real(np.trace(rho_ss_th  @ n_op)))
    td    = trace_distance(rho_ss_num, rho_ss_th)

    return evals, ts, n_t, L, (rho_ss_num, rho_ss_th, n_num, n_th, td)

# ---------- Demo / sanity checks ----------
def demo_qubits():
    print("== One qubit mixed state ==")
    rho1 = one_qubit_mixed_density()
    print(rho1)
    print("Tr(ρ) =", np.trace(rho1).real, "  eigs =", np.round(np.linalg.eigvalsh(rho1), 6))

    print("\n== Two qubits mixed state ==")
    rho2 = two_qubit_mixed_density()
    print(rho2)

    print("\nNumeric expectations vs analytic targets:")
    report = validate_two_qubit_expectations(rho2)
    for k, d in report.items():
        print(f"  {k:10s}  numeric={d['numeric']: .10f}  target={d['target']: .10f}  error={d['error']: .2e}")

    # Reduced density matrices
    rho_A = partial_trace(rho2, (2,2), "B")
    rho_B = partial_trace(rho2, (2,2), "A")
    print("\nReduced states:")
    print("ρ_A =\n", rho_A)
    print("ρ_B =\n", rho_B)

def demo_ho(do_plots=True):
    print("\n== Harmonic oscillator coupled to bath (with analytic thermal state) ==")
    evals, ts, n_t, L, ss = ho_example_run(nmax=10, n0=5, t_final=20.0, num_t=100, do_plots=do_plots)
    rho_ss_num, rho_ss_th, n_num, n_th, td = ss

    print("Liouvillian dim =", L.shape)
    print("A few eigenvalues:", evals[:8])
    print(f"Steady ⟨n⟩: numeric={n_num:.8f}, analytic (τ)={n_th:.8f}")
    print(f"Trace distance(ρ_ss,num , ρ_th) = {td:.3e}")
    if do_plots:
        plt.show()

if __name__ == "__main__":
    demo_qubits()
    demo_ho(do_plots=True)


