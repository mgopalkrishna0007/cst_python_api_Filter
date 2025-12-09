import numpy as np
import matplotlib.pyplot as plt

# Constants
f0 = 79e9  # Center frequency (Hz)
FBW = 0.05  # Fractional bandwidth
BW = f0 * FBW

fmin = 74e9
fmax = 84e9
fstep = 1e6
f = np.arange(fmin, fmax + fstep, fstep)

N = 7  # Filter order

Lr_dB = -30  # Reflection coefficient in dB
Lar = -10 * np.log10(1 - 10**(0.1 * Lr_dB))  # Passband ripple from Lr

# Prototype element calculations (same as before)
g = np.zeros(N + 2)
R = np.zeros((N + 2, N + 2))

g[0] = 1
beta = np.log(np.cosh(Lar / 17.37) / np.sinh(Lar / 17.37))
gamma = np.sinh(beta / (2 * N))
g[1] = 2 * np.sin(np.pi / (2 * N)) / gamma

for i in range(1, N):
    num = 4 * np.sin((2 * i + 1) * np.pi / (2 * N)) * np.sin((2 * i - 1) * np.pi / (2 * N))
    denom = gamma**2 + (np.sin(i * np.pi / N))**2
    g[i + 1] = (1 / g[i]) * (num / denom)

g[-1] = (np.cosh(beta / 4))**2 if N % 2 == 0 else 1

# Coupling matrix (modified for bandstop)
R[0, 1] = 1 / np.sqrt(g[0] * g[1])
R[N, N + 1] = 1 / np.sqrt(g[N] * g[N + 1])
for i in range(1, N):
    R[i, i + 1] = 1 / np.sqrt(g[i] * g[i + 1])

# For bandstop, we need cross-couplings to create transmission zeros at f0
M_coupling = R + R.T

# Add diagonal elements for bandstop response
for i in range(1, N + 1):
    M_coupling[i, i] = -1 / (FBW * (f0 / BW))  # Creates rejection at center frequency

# System matrices
U = np.eye(M_coupling.shape[0])
U[0, 0] = 0
U[-1, -1] = 0

R_mat = np.zeros_like(M_coupling)
R_mat[0, 0] = 1
R_mat[-1, -1] = 1

# Frequency loop (modified for bandstop response)
lambda_vals = (f0 / BW) * (1 / ((f / f0) - (f0 / f)))  # Inverted for bandstop

S11 = np.zeros(len(f), dtype=complex)
S21 = np.zeros(len(f), dtype=complex)

for i, lam in enumerate(lambda_vals):
    A = lam * U - 1j * R_mat + M_coupling
    A_inv = np.linalg.inv(A)
    S11[i] = 1 + 2j * A_inv[0, 0]
    S21[i] = -2j * A_inv[-1, 0]

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
freq_GHz = f / 1e9

axs[0, 0].plot(freq_GHz, 20 * np.log10(np.abs(S11)), linewidth=2)
axs[0, 0].set_title("|S11| (dB)")
axs[0, 0].set_xlabel("Frequency (GHz)")
axs[0, 0].set_ylabel("Magnitude (dB)")
axs[0, 0].axvline(x=f0/1e9, color='r', linestyle='--', alpha=0.5)
axs[0, 0].grid(True)

axs[0, 1].plot(freq_GHz, 20 * np.log10(np.abs(S21)), linewidth=2)
axs[0, 1].set_title("|S21| (dB)")
axs[0, 1].set_xlabel("Frequency (GHz)")
axs[0, 1].set_ylabel("Magnitude (dB)")
axs[0, 1].axvline(x=f0/1e9, color='r', linestyle='--', alpha=0.5)
axs[0, 1].grid(True)

axs[1, 0].plot(freq_GHz, np.abs(S11), linewidth=2)
axs[1, 0].set_title("|S11| (abs)")
axs[1, 0].set_xlabel("Frequency (GHz)")
axs[1, 0].set_ylabel("Magnitude")
axs[1, 0].axvline(x=f0/1e9, color='r', linestyle='--', alpha=0.5)
axs[1, 0].grid(True)

axs[1, 1].plot(freq_GHz, np.abs(S21), linewidth=2)
axs[1, 1].set_title("|S21| (abs)")
axs[1, 1].set_xlabel("Frequency (GHz)")
axs[1, 1].set_ylabel("Magnitude")
axs[1, 1].axvline(x=f0/1e9, color='r', linestyle='--', alpha=0.5)
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()