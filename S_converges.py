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


def compare_sort_vs_epsilon():
    """
    Compare exact S(x) with S_tilde_epsilon(x) for different values of epsilon
    """
    # x is a test vector near to the one shown in figure 2
    x = np.array([0.5, 0.2, 0.4, 0.8, 0.7, 0.3, 0.25, 0.65, 0.85, 0.45])
    n = len(x)
    
    # Uniorm weights
    a = np.ones(n) / n
    b = np.ones(n) / n
    y = np.linspace(0, 1, n)
    
    # Exact sort
    s_exact = np.sort(x)
    
    # Epsilon values to test
    epsilons = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    
    # Compute S_etilde_epsilon for each value
    results = {}
    
    print("Compute of S_tilde_epsilon for epsilon values...\n")
    for eps in epsilons:
        print(f"epsilon = {eps:.0e}")
        r_eps, s_eps = sinkhorn_rank_sort(a, x, b, y, epsilon=eps, verbose=True)
        results[eps] = s_eps
        print()
    
    # ========================================
    # FIGURE : Comparison S(x) vs S_tilde_epsilon(x)
    # ========================================
    positions = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot exact S(x)
    ax.plot(positions, s_exact, marker='o', markersize=10, linewidth=3,
            label='Exact S(x)', color='black', linestyle='-', alpha=0.9)
    
    # Plot S_tilde_epsilon(x)
    markers = ['s', '^', 'D', 'v', 'p']
    linestyles = ['--', '-.', ':', '--', '-.']
    
    for i, eps in enumerate(epsilons):
        ax.plot(positions, results[eps], marker=markers[i], markersize=8, linewidth=2,
                label=f'S_tilde_epsilon(x), epsilon={eps:.0e}', linestyle=linestyles[i], alpha=0.8)
    
    # Personnalization
    ax.set_xlabel('Position (sorted index)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_title('Comparison : Exact sort S(x) vs Sinkhorn sort S_tilde_epsilon(x)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{i+1}' for i in range(n)], fontsize=11)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add a note
    note = f"tested vector : x = {x}"
    ax.text(0.5, -0.12, note, transform=ax.transAxes, 
            fontsize=9, ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('comparison_sort_vs_epsilon_lines.png', dpi=300, bbox_inches='tight')
    print("File saved : comparison_sort_vs_epsilon_lines.png\n")
    plt.show()
    
    # ========================================
    # RECAPITULATIVE TABLE
    # ========================================
    print("="*80)
    print("RECAPITULATIVE TABLE : Sorted values for each epsilon ")
    print("="*80)
    print(f"{'Position':<10} {'Exact S(x)':<15}", end='')
    for eps in epsilons:
        print(f"{'eps = '+f'{eps:.0e}':<15}", end='')
    print()
    print("-"*80)
    
    for i in range(n):
        print(f"{i+1:<10} {s_exact[i]:<15.4f}", end='')
        for eps in epsilons:
            print(f"{results[eps][i]:<15.4f}", end='')
        print()
    
    print("="*80)


# ============================================================
# EXÉCUTION
# ============================================================
if __name__ == "__main__":
    compare_sort_vs_epsilon()
    
