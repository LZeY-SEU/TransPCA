import numpy as np
from pca_utils import *
from sklearn.model_selection import KFold



class TransPCA:
    def __init__(self, target, source_list, cov='covariance', 
                 first_step='GB', r0='eig_ratio', rk_list='eig_ratio', delta=0.1, tau=0.3):
        """
        Initialize the TransPCA model.
        
        Parameters:
        - target (ndarray): The target dataset (shape n0 x p).
        - source_list (list): List of K source datasets, each with shape (nk x p).
        - cov (str): The method to compute the covariance matrix ('covariance' by default). 
                     Future extensions may include 'kendall', etc.
        - first_step (str): Either 'GB' or 'PCA' to decide which method to use for the first step.
        - r0 (str or int): The target subspace dimension ('variance_ratio', 'AIC', 'BIC','eig_ratio', or an integer).
        - rk_list (str or list): Criteria for dimensionality selection for the source datasets 
                            ('AIC', 'BIC', 'variance_ratio','eig_ratio', or a list of integers).
        - delta (float): The threshold for the projection error when selecting the shared subspace.
        - tau (float): The threshold for choosing useful sources.
        """
        self.first_step = first_step
        self.delta = delta  
        self.tau = tau
        self.target = target
        self.source_list = source_list
        self.cov = cov
        
        # 1. Compute covariance matrices from raw datasets.
        if cov == 'covariance':
            self.target_cov = np.cov(target, rowvar=False)  # shape: (p, p)
            self.source_cov_list = [np.cov(source, rowvar=False) for source in source_list]
        elif cov == 'kendall':
            self.target_cov = sample_Kendall_tau(target)  # shape: (p, p)
            self.source_cov_list = [sample_Kendall_tau(source) for source in source_list]
        else:
            raise ValueError("The 'cov' parameter must be either 'covariance' or 'kendall'.")
        
        # 2. Record sample sizes: n0 for target and nk_list for each source.
        self.n0 = target.shape[0]
        self.nk_list = np.array([source.shape[0] for source in source_list])
        
        # 3. Determine r0 (target subspace dimension)
        if isinstance(r0, str):
            self.r0 = select_pca_dimension(self.target_cov, criterion=r0)
        else:
            self.r0 = r0
        
        # 4. Determine rk_list (source subspace dimensions)
        if isinstance(rk_list, str):
            self.rk_list = [select_pca_dimension(cov, criterion=rk_list) for cov in self.source_cov_list]
        else:
            self.rk_list = np.array(rk_list)

        # 5. Compute the target and source principal components using the computed dimensions.
        self.U0 = compute_pca(self.target_cov, self.r0)  # shape: (p x r0)
        self.Uk_list = [compute_pca(cov, rk_val) for cov, rk_val in zip(self.source_cov_list, self.rk_list)]
        self.p = self.U0.shape[0]
        
        # 6. Initialize placeholders for later use.
        self.Us = None      # Shared subspace from GB (shape p x r0)
        self.Urs = None     # Shared subspace after fine-tuning (shape p x rs)
        self.selected_k_list = np.array([1] * len(self.source_cov_list))


    def fit(self, max_iteration=10):
        """
        Fit the TransPCA model using the first step as 'GB'. This is an iterative algorithm.
        
        Parameters:
        - max_iteration (int): The maximum number of iterations for the GB algorithm.
        """
        previous_selected_k_list = self.selected_k_list.copy()  # Store the previous selected_k_list to check for convergence

        for t in range(max_iteration):
            #print(f"Iteration {t + 1}/{max_iteration}")
            
            # 1. Use get_GB_Us or get_PCA_Us to compute the updated Us
            if self.first_step == 'GB':
                self.get_GB_Us()  
            elif self.first_step == 'PCA':
                self.get_PCA_Us()  
            else:
                raise ValueError("first_step must be either 'GB' or 'PCA'.")  
            
            # 2. Use optional_finetuning to update Urs and rs
            self.optional_finetuning()
            
            # 3. Update selected_k_list using select_k
            if self.Urs is None:
                break
            else:
                self.select_k()

            # 4. Check for early stopping: if selected_k_list has not changed
            if np.array_equal(self.selected_k_list, previous_selected_k_list):
                # print("Early stopping: selected_k_list has not changed.")
                break  # Stop the iterations if there's no change in selected_k_list

            # Update previous_selected_k_list for the next iteration
            previous_selected_k_list = self.selected_k_list.copy()

        # Final result after iterations
        # print("Fitting complete.")
        # print("Final selected sources:", self.selected_k_list)
        # print("Final shared subspace dimension (rs):", self.rs)

    def get_PCA_Us(self):
        """
        Apply PCA-based aggregation to the sample covariance matrices.
        """
        # 1. Check if selected_k_list and nk_list have the same length
        if len(self.selected_k_list) != len(self.nk_list):
            raise ValueError("Length of selected_k_list and nk_list must be the same.")
        
        # 2. Calculate weight_list = selected_k_list * nk_list
        weight_list = np.array(self.selected_k_list) * np.array(self.nk_list)
        
        # 3. Initialize Sigma_GB = n0 * target_cov
        Sigma_PCA = self.n0 * self.target_cov
        
        # 4. Add weighted contributions from the selected source datasets
        for k, source_cov in enumerate(self.source_cov_list):
            if self.selected_k_list[k] == 1:  # Only add if this source is selected
                Sigma_PCA += weight_list[k] * source_cov
        
        # 5. Extract the top r0 principal components from Sigma_PCA
        # Perform eigen decomposition to extract the top r0 eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_PCA)
        
        # Sort the eigenvectors based on the eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        # Get the first r0 eigenvectors (as columns)
        self.Us = eigenvectors[:, sorted_indices[:self.r0]]
    

    def get_GB_Us(self):
        """
        Compute the Grassmannian Barycenter (GB) using the selected sources and their weights,
        and extract the first r principal components.
        """
        # 1. Check if selected_k_list and nk_list have the same length
        if len(self.selected_k_list) != len(self.nk_list):
            raise ValueError("Length of selected_k_list and nk_list must be the same.")
        
        # 2. Calculate weight_list = selected_k_list * nk_list
        weight_list = np.array(self.selected_k_list) * np.array(self.nk_list)
        
        # 3. Initialize Sigma_GB = n0 * U0 @ U0.T
        Sigma_GB = self.n0 * self.U0 @ self.U0.T
        
        # 4. Add weighted contributions from the selected source datasets
        for k, Uk in enumerate(self.Uk_list):
            if self.selected_k_list[k] == 1:  # Only add if this source is selected
                Sigma_GB += weight_list[k] * Uk @ Uk.T
        
        # 5. Extract the top r0 principal components from Sigma_GB
        # Perform eigen decomposition to extract the top r0 eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_GB)
        
        # Sort the eigenvectors based on the eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        # Get the first r0 eigenvectors (as columns)
        self.Us = eigenvectors[:, sorted_indices[:self.r0]]
        
        #return self.Us  # Optionally return the computed Us

    def select_k(self):
        """
        Select the source datasets based on the scaled projection error threshold (tau).
    
        - `tau` is now a relative threshold within [0,1] and is scaled to [0, rs].
        - A larger `tau` means **more source datasets are included** in the transfer learning.
        - If `tau = 0`, no source dataset is included.
        - If `tau = 1`, all available sources are included.

        Updates:
        - `self.selected_k_list`: A binary array indicating selected source datasets.
        """
        if self.Urs is None:
            raise ValueError("self.Urs must be set before calling select_k.")

        # Scale tau from [0,1] to [0, rs]
        tau_scaled = self.tau * self.rs if self.rs > 0 else 0  

        selected_k_list = []

        for k, Uk in enumerate(self.Uk_list):
            proj_error = np.trace((np.eye(self.p) - Uk @ Uk.T) @ self.Urs @ self.Urs.T)

            # Use the scaled tau threshold
            if proj_error < tau_scaled:
                selected_k_list.append(1)  # Accept the source dataset
            else:
                selected_k_list.append(0)  # Reject the source dataset

        self.selected_k_list = np.array(selected_k_list)


    def optional_finetuning(self):
        """
        Fine-tune the shared subspace based on the scaled projection error threshold (delta).
    
        - `delta` is a relative threshold within [0,1] and is scaled to [0, r0].
        - A larger `delta` means **more aggressive migration** from source datasets.
        - If `delta = 0`, no migration is performed, and only the target PCA is used.
        - If `delta = 1`, we try to incorporate as much shared information as possible.

        Updates:
        - `self.Urs`: The selected shared subspace.
        - `self.rs`: The dimension of `self.Urs`.
        - `self.finetuned_U0`: The final adjusted target subspace.
        """
        if self.Us is None or self.U0 is None:
            raise ValueError("Us and U0 must be set before finetuning.")

        # Scale delta from [0,1] to [0, r0]
        delta_scaled = self.delta * self.r0

        rs = 0  # Initialize the shared subspace dimension

        for r in range(1, self.r0 + 1):
            Ur = self.Us[:, :r]  # Take the first r components of Us
            proj_error = np.trace((np.eye(self.p) - self.U0 @ self.U0.T) @ Ur @ Ur.T)

            # Use the scaled delta threshold
            if proj_error < delta_scaled:
                rs = r
            else:
                break  # Stop when projection error exceeds threshold

        # If no migration is performed (rs = 0), use only the target PCA
        if rs == 0:
            # print("Using target estimate only (no migration).")
            self.finetuned_U0 = self.U0
            self.Urs = None
            self.rs = 0
        else:
            self.Urs = self.Us[:, :rs]  # Select the top rs components
            self.rs = rs

            # Fine-tune by removing the influence of Urs and recomputing the remaining subspace
            Prs_perp = np.eye(self.p) - self.Urs @ self.Urs.T
            Uf = np.linalg.svd(Prs_perp @ self.target_cov @ Prs_perp)[0][:, :self.r0 - self.rs]
            self.finetuned_U0 = np.hstack((self.Urs, Uf))  

class cvTransPCA(TransPCA):
    def __init__(self, target, source_list, cov='covariance',
                 first_step='GB', r0='eig_ratio', rk_list='eig_ratio',
                 n_folds=5, delta_grid=None, tau_grid=None):
        """
        Cross-validation extension of TransPCA to optimize delta and tau.
        """
        super().__init__(target, source_list, cov=cov,
                         first_step=first_step, r0=r0, rk_list=rk_list)
        self.n_folds = n_folds

        # 设定 delta 和 tau 的搜索网格
        self.delta_grid = np.logspace(-3, 0, num=10) if delta_grid is None else delta_grid
        self.tau_grid = np.logspace(-3, 0, num=10) if tau_grid is None else tau_grid
        
        # 记录最佳参数
        self.selected_delta = None
        self.selected_tau = None

    def fit(self, max_iteration=10):
        """
        Cross-validation to select best delta and tau, then train TransPCA with selected parameters.
        """
        best_metric = float('-inf')
        best_delta = None
        best_tau = None

        # 计算 target 数据整体的 Frobenius norm ||X||_F^2
        total_norm = np.linalg.norm(self.target, 'fro')**2

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for delta_val in self.delta_grid:
            for tau_val in self.tau_grid:
                sum_metric = 0  # 记录所有 fold 的投影误差

                for train_idx, val_idx in kf.split(self.target):
                    X_train, X_val = self.target[train_idx], self.target[val_idx]

                    # 用 TransPCA 训练
                    temp_model = TransPCA(X_train, self.source_list,
                                          cov=self.cov, first_step=self.first_step, r0=self.r0,
                                          rk_list=self.rk_list,delta=delta_val, tau=tau_val)

                    temp_model.fit(max_iteration=10)

                    # 计算投影误差 ||PX_val||_F^2
                    P = temp_model.finetuned_U0 @ temp_model.finetuned_U0.T
                    proj_val = P @ X_val.T
                    sum_metric += np.linalg.norm(proj_val, 'fro')**2

                # 归一化得到保留信息比例
                info_retained = sum_metric / total_norm

                if info_retained > best_metric:
                    best_metric = info_retained
                    best_delta = delta_val
                    best_tau = tau_val

        # 记录选出的最佳 delta 和 tau
        self.selected_delta = best_delta
        self.selected_tau = best_tau
        # print(f"Selected delta: {best_delta:.4f}, Selected tau: {best_tau:.4f}, Best information retention: {best_metric:.4f}")

        # 用选出的参数重新训练 TransPCA
        self.delta = self.selected_delta
        self.tau = self.selected_tau
        super().fit(max_iteration=max_iteration)
