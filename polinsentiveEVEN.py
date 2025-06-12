import numpy as np
import matplotlib.pyplot as plt

def create_symmetric_pattern_even(m):
    if m % 2 != 0:
        raise ValueError("Matrix size must be even.")
    
    matrix = np.zeros((m, m), dtype=int)
    half = m // 2

    # Fill lower triangle of upper-left quadrant (i >= j)
    for i in range(half):
        for j in range(i + 1):  # includes diagonal
            val = np.random.randint(0, 2)
            matrix[i, j] = val

            # Symmetric mirrors
            matrix[j, i] = val                        # mirror across main diagonal in upper-left
            matrix[i, m - 1 - j] = val                # mirror to upper-right
            matrix[j, m - 1 - i] = val                # symmetric in upper-right
            matrix[m - 1 - i, j] = val                # mirror to bottom-left
            matrix[m - 1 - j, i] = val                # symmetric in bottom-left
            matrix[m - 1 - i, m - 1 - j] = val        # mirror to bottom-right
            matrix[m - 1 - j, m - 1 - i] = val        # symmetric in bottom-right

    return matrix

# Parameters
m = 20  # even dimension
dimension = 10  # physical dimension in cm

# Generate pattern
pattern = create_symmetric_pattern_even(m)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
cmap = plt.cm.Reds
cmap.set_under('white')

c = ax.imshow(pattern, cmap=cmap, vmin=0.01, extent=[0, dimension, 0, dimension])
ax.set_title(f"{m}x{m} Symmetric Pattern from Lower Triangle", fontsize=14)
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_xticks(np.linspace(0, dimension, m + 1))
ax.set_yticks(np.linspace(0, dimension, m + 1))
ax.grid(True, which='both', color='gray', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.show()
