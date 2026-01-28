import numpy as np
import matplotlib.pyplot as plt

def normalize_and_squash(x):
    """Implementation of g_tilde(x) of equation (4)"""
    n = len(x)
    mean_x = np.mean(x)
    x_centered = x - mean_x
    std_x = np.linalg.norm(x_centered) / np.sqrt(n)
    
    if std_x < 1e-10:
        return np.ones(n) * 0.5
    
    x_standardized = x_centered / std_x
    x_tilde = (np.arctan(x_standardized) + np.pi/2) / np.pi
    
    return x_tilde


def sinkhorn_rank_sort(a, x, b, y, epsilon, eta=1e-6, max_iter=1000, h=lambda z: z**2, verbose=False):
    n = len(x)
    m = len(y)
    
    # Normalization
    x_tilde = normalize_and_squash(x)
    
    # Cost matrix
    C = h(y[None, :] - x_tilde[:, None])
    
    # Kernel matrix K = exp(-C/ε)
    K = np.exp(-C / epsilon)
    
    # Initialization
    u = np.ones(n)
    v = np.ones(m)
    
    # Sinkhorn Iterations
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # Alternated updates
        v = b / (K.T @ u + 1e-300)  # Made to avoid division by 0
        u = a / (K @ v + 1e-300)
        
        # Check if it converges every 10 steps
        if (iteration + 1) % 10 == 0:
            P = u[:, None] * K * v[None, :]
            error = np.linalg.norm(P.T @ np.ones(n) - b)
            
            if error < eta:
                if verbose:
                    print(f"Sinkhorn converges at iteration {iteration+1}, error = {error:.2e}")
                break
            
            u_change = np.linalg.norm(u - u_old) / (np.linalg.norm(u) + 1e-10)
            if u_change < 1e-8:
                if verbose:
                    print(f"Sinkhorn converges (u_change < 1e-8) at iteration {iteration+1}")
                break
    
    # Final transport matrix
    P = u[:, None] * K * v[None, :]
    
    # b_bar : cumulative b
    b_bar = np.cumsum(b)
    
    # Rank and Sort operators
    r_eps = n * (P @ b_bar) / a
    s_eps = (P.T @ x) / b
    
    return r_eps, s_eps


def exact_sort(x):
    """Exact sort S(x)"""
    return np.sort(x)


def numerical_gradient(func, x, idx, h=1e-5):
    """
    Compute the numerical gradient of func related to x[idx]
    using the finite centered differences
    """
    n = len(x)
    e = np.zeros(n)
    e[idx] = 1.0
    
    f_plus = func(x + h * e)
    f_minus = func(x - h * e)
    
    gradient = (f_plus - f_minus) / (2 * h)
    
    return gradient


def compute_all_gradients(x, epsilon=None, h=1e-5, use_sinkhorn=True):
    """
    Compute all gradients for all i,j in [1;n]
    
    Returns gradient_matrix, a matrix n x n
    """
    n = len(x)
    gradient_matrix = np.zeros((n, n))
    
    if use_sinkhorn:
        # Gradients of S_tilde_epsilon(x)
        for i in range(n):
            for j in range(n):
                def func_j(x_perturbed):
                    a = np.ones(n) / n
                    b = np.ones(n) / n
                    y = np.linspace(0, 1, n)
                    _, s_eps = sinkhorn_rank_sort(a, x_perturbed, b, y, epsilon)
                    return s_eps[j]
                
                gradient_matrix[i, j] = numerical_gradient(func_j, x, i, h)
    else:
        # Gradients of exact S(x)
        for i in range(n):
            for j in range(n):
                def func_j(x_perturbed):
                    s = exact_sort(x_perturbed)
                    return s[j]
                
                gradient_matrix[i, j] = numerical_gradient(func_j, x, i, h)
    
    return gradient_matrix


def verify_differentiability():
    """
    Verify our second assertion (differentiability of S_tilde_epsilon at the contrary of S)
    """
    
    # Test vector
    x = np.array([0.5, 0.2, 0.4, 0.8, 0.7, 0.3, 0.25, 0.65, 0.85, 0.45])
    n = len(x)
    
    print("="*80)
    print("VERIFICATION OF THE DIFFERENTIABILITY")
    print("="*80)
    print(f"Tested vector : x = {x}\n")
    
    threshold = 1e-6
    
    # ========================================
    # 1. Exact sort gradients S(x)
    # ========================================
    print("Computation of gradients of exact S(x)...")
    
    grad_exact = compute_all_gradients(x, use_sinkhorn=False, h=1e-5)
    
    # Count the number of non zeros
    non_zero_exact = np.abs(grad_exact) > threshold
    count_non_zero_exact = np.sum(non_zero_exact)
    total_gradients = n * n
    percent_non_zero_exact = 100 * count_non_zero_exact / total_gradients
    
    print("RESULTS FOR EXACT S(x) :")
    print("-" * 80)
    print(f"Non zeros gradients : {count_non_zero_exact} / {total_gradients} ({percent_non_zero_exact:.1f}%)")
    print(f"Zeros gradients     : {total_gradients - count_non_zero_exact} / {total_gradients} ({100-percent_non_zero_exact:.1f}%)\n")
    
    # ========================================
    # 2. Gradients of Sinkhorn S_tilde_epsilon(x)
    # ========================================
    epsilons = [1e-2, 1e-3]
    results_sinkhorn = {}
    
    for eps in epsilons:
        print(f"Computation of gradients of S_tilde_epsilon(x) with epsilon ={eps:.0e}...")
        
        grad_sinkhorn = compute_all_gradients(x, epsilon=eps, use_sinkhorn=True, h=1e-5)
        
        non_zero_sinkhorn = np.abs(grad_sinkhorn) > threshold
        count_non_zero_sinkhorn = np.sum(non_zero_sinkhorn)
        percent_non_zero_sinkhorn = 100 * count_non_zero_sinkhorn / total_gradients
        
        results_sinkhorn[eps] = {
            'matrix': grad_sinkhorn,
            'count': count_non_zero_sinkhorn,
            'percent': percent_non_zero_sinkhorn
        }
        
        print(f"RESULTS FOR S_tilde_epsilon(x) with epsilon={eps:.0e} :")
        print("-" * 80)
        print(f"Non zeros gradients : {count_non_zero_sinkhorn} / {total_gradients} ({percent_non_zero_sinkhorn:.1f}%)")
        print(f"Zeros gradients     : {total_gradients - count_non_zero_sinkhorn} / {total_gradients} ({100-percent_non_zero_sinkhorn:.1f}%)\n")
    
    # ========================================
    # VISUALISATION : Zeros gradients values against non-zero ones 
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    masks = [
        (non_zero_exact, 'Exact S(x)', percent_non_zero_exact),
        (np.abs(results_sinkhorn[epsilons[0]]['matrix']) > threshold, 
         f'S_tilde_epsilon(x), epsilon ={epsilons[0]:.0e}', results_sinkhorn[epsilons[0]]['percent']),
        (np.abs(results_sinkhorn[epsilons[1]]['matrix']) > threshold, 
         f'S_tilde_epsilon(x), epsilon ={epsilons[1]:.0e}', results_sinkhorn[epsilons[1]]['percent'])
    ]
    
    for idx, (mask, title, pct) in enumerate(masks):
        ax = axes[idx]
        im = ax.imshow(mask.astype(int), cmap='gray_r', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'{title}\n{pct:.1f}% non-zeros', fontsize=13, fontweight='bold')
        ax.set_xlabel('Output j', fontsize=11)
        ax.set_ylabel('Input i', fontsize=11)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f'{i+1}' for i in range(n)])
        ax.set_yticklabels([f'{i+1}' for i in range(n)])
        
        ax.set_xticks(np.arange(n) - 0.5, minor=True)
        ax.set_yticks(np.arange(n) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.suptitle('Masks of non-zeros gradients (White = non-zero, Black = zero)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_masks.png', dpi=300, bbox_inches='tight')
    print("File saved : gradient_masks.png\n")
    plt.show()

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    verify_differentiability()
