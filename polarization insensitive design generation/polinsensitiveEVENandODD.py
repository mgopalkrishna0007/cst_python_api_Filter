import numpy as np
import matplotlib.pyplot as plt

def generate_1_8th_shape_odd(m):
    """Generate 1/8th triangle for odd-sized matrix"""
    if m % 2 == 0:
        raise ValueError("Matrix size must be odd.")
    c = m // 2
    shape = np.zeros((m, m), dtype=int)
    # Fill 1/8th triangle including the center row and column
    for i in range(c + 1):  # include center
        for j in range(i, c + 1):
            shape[i, j] = np.random.randint(0, 2)
    return shape

def mirror_8_fold_odd(base):
    """Mirror 8-fold for odd-sized matrix"""
    m = base.shape[0]
    c = m // 2
    matrix = np.zeros_like(base)
    # Mirror the values across 8 symmetric directions
    for i in range(c + 1):  # include center
        for j in range(i, c + 1):
            val = base[i, j]
            # 8 symmetric positions
            coords = [
                (i, j), (j, i),
                (i, m - 1 - j), (j, m - 1 - i),
                (m - 1 - i, j), (m - 1 - j, i),
                (m - 1 - i, m - 1 - j), (m - 1 - j, m - 1 - i)
            ]
            for x, y in coords:
                matrix[x, y] = val
    return matrix

def generate_1_8th_shape_even(m):
    """Generate 1/8th triangle for even-sized matrix"""
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
    """Mirror 8-fold for even-sized matrix"""
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

def generate_symmetric_pattern(n):
    """Main function to generate pattern based on n (odd or even)"""
    
    if n % 2 == 1:  # Odd n
        print(f"n = {n} is odd, using odd matrix algorithm")
        print("="*50)
        
        # Generate 1/8th shape
        shape_1_8 = generate_1_8th_shape_odd(n)
        full_shape = mirror_8_fold_odd(shape_1_8)
        
        # Print matrices
        print("1/8th Triangle Matrix (Odd):")
        print(shape_1_8)
        print("\nFinal Matrix Configuration (Odd):")
        print(full_shape)
        
        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(shape_1_8, cmap='Reds', vmin=0.01)
        axs[0].set_title(f'Random 1/8th Region (Upper-Left Triangle) - Size {n}x{n}')
        axs[0].axis('off')
        
        axs[1].imshow(full_shape, cmap='Reds', vmin=0.01)
        axs[1].set_title(f'Full 8-Fold Symmetric Pattern (Odd) - Size {n}x{n}')
        axs[1].axis('off')
        
    else:  # Even n
        print(f"n = {n} is even, using even matrix algorithm")
        print("="*50)
        
        # Generate 1/8th shape
        shape_1_8 = generate_1_8th_shape_even(n)
        full_shape = mirror_8_fold_even(shape_1_8)
        
        # Print matrices
        print("1/8th Triangle Matrix (Even):")
        print(shape_1_8)
        print("\nFinal Matrix Configuration (Even):")
        print(full_shape)
        
        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(shape_1_8, cmap='Reds', vmin=0.01)
        axs[0].set_title(f'Random 1/8th Region (Lower Triangle) - Size {n}x{n}')
        axs[0].axis('off')
        
        axs[1].imshow(full_shape, cmap='Reds', vmin=0.01)
        axs[1].set_title(f'Full 8-Fold Symmetric Pattern (Even) - Size {n}x{n}')
        axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return shape_1_8, full_shape

# Example usage with different values of n
if __name__ == "__main__":
    # Test with odd n
    print("Testing with odd n = 21:")
    shape_1_8_odd, full_shape_odd = generate_symmetric_pattern(21)
    
    print("\n" + "="*70 + "\n")
    
    # Test with even n
    print("Testing with even n = 20:")
    shape_1_8_even, full_shape_even = generate_symmetric_pattern(20)
