import numpy as np
import matplotlib.pyplot as plt

def generate_lower_triangle_shape_even(m):
    if m % 2 != 0:
        raise ValueError("Matrix size must be even.")
    
    shape = np.zeros((m, m), dtype=int)
    half = m // 2

    # Fill only lower triangle (including diagonal) of upper-left quadrant
    for i in range(half):
        for j in range(i + 1):  # includes diagonal
            shape[i, j] = np.random.randint(0, 2)
    return shape

def mirror_8_fold_even(base):
    m = base.shape[0]
    half = m // 2
    matrix = np.zeros_like(base)

    for i in range(half):
        for j in range(i + 1):
            val = base[i, j]
            matrix[i, j] = val
            matrix[j, i] = val
            matrix[i, m - 1 - j] = val
            matrix[j, m - 1 - i] = val
            matrix[m - 1 - i, j] = val
            matrix[m - 1 - j, i] = val
            matrix[m - 1 - i, m - 1 - j] = val
            matrix[m - 1 - j, m - 1 - i] = val

    return matrix

# Parameters
m = 20  # even size

# Generate random 1/8th region (lower triangle of upper-left quadrant)
shape_1_8_even = generate_lower_triangle_shape_even(m)
full_shape_even = mirror_8_fold_even(shape_1_8_even)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(shape_1_8_even, cmap='Reds', vmin=0.01)
axs[0].set_title('Random 1/8th Region (Lower Triangle, Even Size)')
axs[0].axis('off')

axs[1].imshow(full_shape_even, cmap='Reds', vmin=0.01)
axs[1].set_title('Full 8-Fold Symmetric Pattern (Even Size)')
axs[1].axis('off')

plt.tight_layout()
plt.show()
