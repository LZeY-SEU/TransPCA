import numpy as np  # replacing cupy with numpy
from sklearn.decomposition import PCA

def select_pca_dimension(cov_matrix, criterion='eig_ratio', variance_ratio=0.95, r_max=None):
    """
    Select the number of principal components based on different criteria:
    - 'variance_ratio': Based on cumulative explained variance (default is 95%).
    - 'AIC': Akaike Information Criterion for model selection.
    - 'BIC': Bayesian Information Criterion for model selection.
    - 'eig_ratio': Selects the dimension where the eigenvalue ratio λ_i / λ_{i+1} is maximized.

    Parameters:
    - cov_matrix (ndarray): The covariance matrix of shape (p, p).
    - criterion (str): One of 'variance_ratio', 'AIC', 'BIC', or 'eig_ratio'.
    - variance_ratio (float): The proportion of total variance to be retained when using 'variance_ratio' (default is 0.95).
    - r_max (int, optional): The maximum dimension considered for 'eig_ratio'. Defaults to p//4 if None.

    Returns:
    - r (int): The number of principal components selected based on the specified criterion.
    """
    # Perform PCA on the covariance matrix
    pca = PCA()
    pca.fit(cov_matrix)  # Removed cp.asnumpy() since we're using numpy
    
    # Eigenvalues of the covariance matrix
    eigenvalues = np.array(pca.explained_variance_) 
    n, p = cov_matrix.shape  # Number of samples and features
    
    if criterion == 'variance_ratio':
        # Cumulative variance explained by components
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        r = np.where(cumulative_variance >= variance_ratio)[0][0] + 1  # +1 to match index
    
    elif criterion == 'AIC' or criterion == 'BIC':
        # Case when p < n (low-dimensional case)
        if p < n:
            def calc_bar_ell(j):
                return np.mean(eigenvalues[j+1:])
            
            AIC_values, BIC_values = [], []
            for j in range(p - 1):
                bar_ell = calc_bar_ell(j)
                sum_log_ell = np.sum(np.log(eigenvalues[j+1:]))
                
                AIC_j = (p - j - 1) * np.log(bar_ell) - sum_log_ell - ((p - j - 1) * (p - j + 2)) / n
                BIC_j = (p - j - 1) * np.log(bar_ell) - sum_log_ell - ((p - j - 1) * (p - j + 2)) / (2 * n) * np.log(n)
                
                AIC_values.append(AIC_j)
                BIC_values.append(BIC_j)

            r = np.argmin(AIC_values) + 1 if criterion == 'AIC' else np.argmin(BIC_values) + 1
        
        # Case when p >= n (high-dimensional case)
        else:
            def calc_bar_ell(j):
                return np.mean(eigenvalues[j+1:n])
            
            AIC_values, BIC_values = [], []
            for j in range(n - 1):
                bar_ell = calc_bar_ell(j)
                sum_log_ell = np.sum(np.log(eigenvalues[j+1:n]))
                
                AIC_j = (n - 1 - j) * np.log(bar_ell) - sum_log_ell - ((n - j - 2) * (n - j + 1)) / p
                BIC_j = (n - 1 - j) * np.log(bar_ell) - sum_log_ell - ((n - j - 2) * (n - j + 1)) / (2 * p) * np.log(p)
                
                AIC_values.append(AIC_j)
                BIC_values.append(BIC_j)

            r = np.argmin(AIC_values) + 1 if criterion == 'AIC' else np.argmin(BIC_values) + 1

    elif criterion == 'eig_ratio':
        # Define r_max if not provided
        if r_max is None:
            r_max = p // 4
        r_max = min(r_max, len(eigenvalues) - 1)  # Ensure r_max is within bounds
        
        # Compute eigenvalue ratios λ_i / λ_{i+1}
        eig_ratios = eigenvalues[:-1] / eigenvalues[1:]

        # Select r where the eigenvalue ratio is maximized, constrained by r_max
        r = np.argmax(eig_ratios[:r_max]) + 1  # +1 to match indexing

    else:
        raise ValueError("Invalid criterion. Choose 'variance_ratio', 'AIC', 'BIC', or 'eig_ratio'.")
    
    return int(r) 


def compute_pca(cov_matrix, r):
    """
    Compute PCA on a covariance matrix by eigen decomposition and return the first r principal components.
    
    Parameters:
    - cov_matrix (ndarray): A covariance matrix (shape p x p).
    - r (int): The number of principal components to retain.
    
    Returns:
    - U (ndarray): The first r principal components (shape p x r).
    """
    # Eigen decomposition (since cov_matrix is symmetric)
    U = np.linalg.svd(cov_matrix)[0][:, :r]
    return U


def sample_Kendall_tau(X):
    """
    Compute the sample Kendall tau matrix for a dataset X.
    
    Parameters:
    - X (ndarray): Data matrix of shape (n, p), where n is the number of samples and p is the number of features.
    
    Returns:
    - K (ndarray): The sample Kendall tau matrix of shape (p, p).
    """
    n, p = X.shape
    TK = np.zeros((p, p)) 
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = X[i, :].reshape(p, 1) - X[j, :].reshape(p, 1)
            norm_sq = np.sum(diff ** 2)
            if norm_sq > 0:
                TK += diff @ diff.T / norm_sq
    K = (2 / (n * (n - 1))) * TK
    return K