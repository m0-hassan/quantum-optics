"""
Quantum State Tomography Analysis Script
Calculates normalized two-photon density matrices, traces, entropies,
concurrences, tangles, partial traces, fidelities with Bell states,
and generates 3D bar plots of Re(rho) for each trial.
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Define Pauli matrices for convenience
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

# Define two-qubit Bell state vectors
zero = np.array([1,0], dtype=complex)
one = np.array([0,1], dtype=complex)
Phi_plus = (np.kron(zero, zero) + np.kron(one, one)) / np.sqrt(2)
Phi_minus = (np.kron(zero, zero) - np.kron(one, one)) / np.sqrt(2)
Psi_plus = (np.kron(zero, one) + np.kron(one, zero)) / np.sqrt(2)
Psi_minus = (np.kron(zero, one) - np.kron(one, zero)) / np.sqrt(2)

# List of Bell state density matrices
bell_states = {
    'Phi+': np.outer(Phi_plus, np.conjugate(Phi_plus)),
    'Phi-': np.outer(Phi_minus, np.conjugate(Phi_minus)),
    'Psi+': np.outer(Psi_plus, np.conjugate(Psi_plus)),
    'Psi-': np.outer(Psi_minus, np.conjugate(Psi_minus))
}

def compute_concurrence(rho):
    """Compute the concurrence of a two-qubit density matrix rho."""
    # Pauli Y (sigma_y) for spin-flip operation
    sigma_y = np.array([[0,-1j],[1j,0]])
    # Spin-flipped state: rho_tilde = (sigma_y ⊗ sigma_y) rho* (sigma_y ⊗ sigma_y)
    sy_sy = np.kron(sigma_y, sigma_y)
    rho_tilde = sy_sy @ rho.conj() @ sy_sy
    # Compute eigenvalues of rho * rho_tilde
    eigvals = LA.eigvals(rho @ rho_tilde)
    eigvals = np.real(eigvals)  # keep real part to avoid numerical issues
    # Sort eigenvalues in descending order
    eigvals_sorted = np.sort(eigvals)[::-1]
    # Concurrence formula
    sqrt_vals = np.sqrt(np.maximum(eigvals_sorted, 0))
    C = max(0, sqrt_vals[0] - sqrt_vals[1] - sqrt_vals[2] - sqrt_vals[3])
    return C

def compute_von_neumann_entropy(rho):
    """Compute the von Neumann entropy (base 2) of density matrix rho."""
    eigvals = LA.eigvals(rho)
    eigvals = eigvals[np.abs(eigvals) > 1e-12]  # avoid log(0)
    S = -np.sum(eigvals * np.log2(eigvals))
    return np.real(S)

def partial_trace(rho, subsystem):
    """Compute partial trace: subsystem=0 traces out qubit A (gives rho_B),
    subsystem=1 traces out qubit B (gives rho_A)."""
    rho_tensor = rho.reshape(2,2,2,2)
    if subsystem == 0:
        # Trace out qubit A (indices 0 and 2)
        return np.trace(rho_tensor, axis1=0, axis2=2)
    elif subsystem == 1:
        # Trace out qubit B (indices 1 and 3)
        return np.trace(rho_tensor, axis1=1, axis2=3)
    else:
        raise ValueError("subsystem must be 0 or 1")

def fidelity_with_state(rho, psi):
    """Compute fidelity between density matrix rho and pure state |psi>."""
    return np.real(np.vdot(psi, rho @ psi))

# Example trial density matrices (replace with measured tomography results)
rho_trials = {
    'Trial1': bell_states['Phi+'],  # example pure Bell state
    'Trial2': 0.9*bell_states['Psi-'] + 0.1*np.eye(4)/4  # example mixed state
}

for name, raw_rho in rho_trials.items():
    # Normalize density matrix to trace 1
    rho = raw_rho / np.trace(raw_rho)
    print(f"=== {name} ===")
    print(f"Trace = {np.trace(rho):.3f}")
    # Compute properties
    S = compute_von_neumann_entropy(rho)
    C = compute_concurrence(rho)
    tau = C**2
    print(f"Von Neumann entropy S(ρ) = {S:.4f}")
    print(f"Concurrence C = {C:.4f}, Tangle τ = {tau:.4f}")
    # Partial traces
    rho_A = partial_trace(rho, subsystem=1)
    rho_B = partial_trace(rho, subsystem=0)
    print(f"ρ_A (trace out B) =\n{rho_A}")
    print(f"ρ_B (trace out A) =\n{rho_B}")
    # Fidelities with Bell states
    for bell_name, bell_rho in bell_states.items():
        F = np.real(np.trace(rho @ bell_rho))
        print(f"Fidelity with |{bell_name}> = {F:.4f}")
    print()
    # 3D bar plot of Re(rho)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(rho.shape[0])
    Y = np.arange(rho.shape[1])
    X, Y = np.meshgrid(X, Y)
    x = X.ravel()
    y = Y.ravel()
    z = np.zeros_like(x)
    dx = dy = 0.5
    dz = np.real(rho).ravel()
    ax.bar3d(x, y, z, dx, dy, dz, shade=True)
    ax.set_xlabel('Index i')
    ax.set_ylabel('Index j')
    ax.set_zlabel('Re(ρ_{ij})')
    ax.set_title(f"Re(ρ) for {name}")
    plt.savefig(f"{name}_Re_rho.png")
    plt.close()
