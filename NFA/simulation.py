import numpy as np
from scipy.stats import ortho_group


def apply_g(Ux, g_type):
    if g_type == "linear":
        return Ux
    elif g_type == "quadratic":
        return Ux ** 2
    elif g_type == "cubic":
        return Ux ** 3
    elif g_type == "log":
        return np.log1p(np.abs(Ux))  # Avoid log(0)
    elif g_type == "exp":
        return np.exp(Ux)
    elif g_type == "mix":
        return 0.5 * Ux**2 + 0.3 * np.log1p(np.abs(Ux)) + 0.2 * Ux
    else:
        raise ValueError("Unsupported g function")
    
def generate_multisource_data(K, n, p, rs, rp, g_types="linear", random_state=None):
    """
    Generate multi-source datasets, each with shared and private subspaces
    
    Parameters:
        K: Number of sources
        n: Data size per source
        p: Feature dimension
        rs: Shared subspace dimension
        rp: Private subspace dimension
        g_types: Type of nonlinear function (str or list, if list must have length K)
        random_state: Random seed
        
    Returns:
        datasets: List of (Xk, yk) tuples for K sources
        (Us, Up0): Shared and private subspaces of the first dataset (as target)
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Process g_types parameter
    if isinstance(g_types, str):
        g_types = [g_types] * K
    elif len(g_types) != K:
        raise ValueError("Length of g_types must equal K")
    
    # 1. Generate shared subspace (same for all sources)
    U = ortho_group.rvs(dim=p)
    Us = U[:, :rs]  # Shared subspace
    
    # 2. Generate private subspace for each source
    datasets = []
    Up0 = None  # Save the private subspace of the first dataset
    
    for k in range(K):
        # Generate private subspace
        Up_same = U[:, rs:]  # Part orthogonal to the shared subspace
        Up_rand = ortho_group.rvs(dim=p-rs)
        Up = np.dot(Up_same, Up_rand[:, :rp])  # Final private subspace
        
        # If it's the first dataset, save Up0
        if k == 0:
            Up0 = Up.copy()
        
        # Combine into complete projection matrix
        Uk = np.hstack([Us, Up])
        
        # Generate data
        Xk = np.random.randn(n, p)  # Standard Gaussian input
        Ux = Xk @ Uk  # Low-dimensional projection
        
        # Compute target variable (using corresponding g_type)
        yk = apply_g(Ux, g_types[k])
        yk = yk.mean(axis=1, keepdims=True)  # Take mean as scalar target
        
        datasets.append((Xk, yk))
    
    return datasets, (Us, Up0)