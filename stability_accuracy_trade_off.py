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


def plot_convergence_vs_epsilon():
    """
    Plot a figure showing how R_epsilon_tilde and S_epsilon_tilde evolve with epsilon
    """
    # x is a test vector near to the one shown in figure 2
    x = np.array([0.5, 0.2, 0.4, 0.8, 0.7, 0.3, 0.25, 0.65, 0.85, 0.45])
    n = len(x)
    
    # Uniorm weights
    a = np.ones(n) / n
    b = np.ones(n) / n
    y = np.linspace(0, 1, n)
    
    # Exact operators or rank and sorting
    r_exact = np.argsort(np.argsort(x)) + 1
    s_exact = np.sort(x)
    
    # Epsilon gam
    epsilons = np.logspace(-4, 1, 20)
    
    # Stockage of the results
    r_eps_all = []
    s_eps_all = []
    
    print("Computation for different epsilon values...")
    for i, eps in enumerate(epsilons):
        print(f"epsilon = {eps:.1e} ({i+1}/{len(epsilons)})")
        
        try:
            r_eps, s_eps = sinkhorn_rank_sort(a, x, b, y, epsilon=eps, eta=1e-5, max_iter=2000)
            
            r_eps_all.append(r_eps)
            s_eps_all.append(s_eps)
            
        except Exception as e:
            print(f"Errors for epsilon = {eps:.1e}: {e}")
            r_eps_all.append(np.nan * np.ones(n))
            s_eps_all.append(np.nan * np.ones(n))
    
    r_eps_all = np.array(r_eps_all)
    s_eps_all = np.array(s_eps_all)
    
    # ========================================
    # FIGURE : R and S in funtion of epsilon
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Ranks
    ax = axes[0]
    for i in range(n):
        ax.semilogx(epsilons, r_eps_all[:, i], marker='o', markersize=4, 
                    label=f'x[{i}]={x[i]:.2f}', alpha=0.7)
        ax.axhline(r_exact[i], color=f'C{i}', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('epsilon (regularization)', fontsize=12)
    ax.set_ylabel('Sinkhorn rank R_tilde_epsilon', fontsize=12)
    ax.set_title('Convergence of rank in funtion of epsilon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=2, loc='best')
    
    # Subplot 2: Sorting
    ax = axes[1]
    for j in range(n):
        ax.semilogx(epsilons, s_eps_all[:, j], marker='s', markersize=4,
                    label=f'Position {j+1}', alpha=0.7)
        ax.axhline(s_exact[j], color=f'C{j}', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('epsilon (regularization)', fontsize=12)
    ax.set_ylabel('Sinkhorn sort S_tilde_epsilon', fontsize=12)
    ax.set_title('Convergence of sort in funtion of epsilon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, ncol=2, loc='best')
    
    plt.tight_layout()
    plt.savefig('convergence_vs_epsilon.png', dpi=300, bbox_inches='tight')
    print("File saved : convergence_vs_epsilon.png")
    plt.show()

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    plot_convergence_vs_epsilon()