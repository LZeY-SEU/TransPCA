import numpy as np  # replacing cupy with numpy
from pca_utils import *  # Note: You may need to modify this import if it contains CuPy-specific code
from sklearn.model_selection import KFold


class TransPCA:
    def __init__(self, target, source_list, cov='covariance', 
                 first_step='GB', r0='eig_ratio', rk_list='eig_ratio', delta=0.1, tau=0.3, r0s=None, n0=None, nk_list=None):
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
        - r0s (int): the shared subspace dimension (if not None, overwrite the rs chosen by delta).
        """
        self.first_step = first_step
        self.delta = delta  
        self.tau = tau
        self.target = np.asarray(target)
        self.source_list = [np.asarray(source) for source in source_list]
        self.cov = cov
        self.r0s = r0s
        
        # 1. Record sample sizes: n0 for target and nk_list for each source.
        if n0 is not None:
            self.n0 = n0
        else:
            self.n0 = self.target.shape[0]
        
        if nk_list is not None:
            self.nk_list = np.array(nk_list)
        else:
            self.nk_list = np.array([source.shape[0] for source in self.source_list])

        # 2. Compute covariance matrices from raw datasets.
        if cov == 'covariance':
            self.target_cov = (self.target.T @ self.target) / self.target.shape[0]
            self.source_cov_list = [(source.T @ source) / source.shape[0] for source in self.source_list]
        elif cov == 'kendall':
            self.target_cov = np.asarray(sample_Kendall_tau(self.target)) 
            self.source_cov_list = [np.asarray(sample_Kendall_tau(source)) for source in self.source_list]
        else:
            raise ValueError("The 'cov' parameter must be either 'covariance' or 'kendall'.")
        
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
        self.U0 = compute_pca(self.target_cov, self.r0) 
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
        previous_selected_k_list = self.selected_k_list.copy() 
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
                #print("Early stopping: selected_k_list has not changed.")
                break  # Stop the iterations if there's no change in selected_k_list

            # Update previous_selected_k_list for the next iteration
            previous_selected_k_list = self.selected_k_list.copy()


    def get_PCA_Us(self):
        """
        Apply PCA-based aggregation to the sample covariance matrices.
        """
        # 1. Check if selected_k_list and nk_list have the same length
        if len(self.selected_k_list) != len(self.nk_list):
            raise ValueError("Length of selected_k_list and nk_list must be the same.")
        
        # 2. Calculate weight_list = selected_k_list * nk_list
        weight_list = self.selected_k_list * self.nk_list
        
        # 3. Initialize Sigma_GB = n0 * target_cov
        Sigma_PCA = self.n0 * self.target_cov
        
        # 4. Add weighted contributions from the selected source datasets
        for k, source_cov in enumerate(self.source_cov_list):
            #if self.selected_k_list[k] == 1:  # Only add if this source is selected
            Sigma_PCA += weight_list[k] * source_cov
        
        # 5. Extract the top r0 principal components from Sigma_PCA
        # Perform eigen decomposition to extract the top r0 eigenvectors
        self.Us = np.linalg.svd(Sigma_PCA)[0][:, :self.r0]
    

    def get_GB_Us(self):
        """
        Compute the Grassmannian Barycenter (GB) using the selected sources and their weights,
        and extract the first r principal components.
        """
        # 1. Check if selected_k_list and nk_list have the same length
        if len(self.selected_k_list) != len(self.nk_list):
            raise ValueError("Length of selected_k_list and nk_list must be the same.")
        
        # 2. Calculate weight_list = selected_k_list * nk_list
        weight_list = self.selected_k_list * self.nk_list
        
        # 3. Initialize Sigma_GB = n0 * U0 @ U0.T
        Sigma_GB = self.n0 * self.U0 @ self.U0.T
        
        # 4. Add weighted contributions from the selected source datasets
        for k, Uk in enumerate(self.Uk_list):
            #if self.selected_k_list[k] == 1:  # Only add if this source is selected
            Sigma_GB += weight_list[k] * Uk @ Uk.T
        
        # 5. Extract the top r0 principal components from Sigma_GB
        # Perform eigen decomposition to extract the top r0 eigenvectors
        self.Us = np.linalg.svd(Sigma_GB)[0][:, :self.r0]
        

    def select_k(self):
        """
        Select the source datasets based on the scaled projection error threshold (tau).
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
        if self.r0s is not None: #overwrite
            self.Urs = self.Us[:, :self.r0s]  # Select the top rs components
            self.rs = self.r0s
            
            # Fine-tune by removing the influence of Urs and recomputing the remaining subspace
            Prs_perp = np.eye(self.p) - self.Urs @ self.Urs.T
            Uf = np.linalg.svd(Prs_perp @ self.target_cov @ Prs_perp)[0][:, :self.r0 - self.rs]
            self.finetuned_U0 = np.hstack((self.Urs, Uf))  

        elif rs == 0:
            #print("Using target estimate only (no migration).")
            self.finetuned_U0 = self.U0
            self.Urs = None
            self.rs = 0
        else:
            self.Urs = self.Us[:, :rs]  # Select the top rs components
            self.rs = rs

            # Fine-tune by removing the influence of Urs and recomputing the remaining subspace
            Prs_perp = np.eye(self.p) - self.Urs @ self.Urs.T
            self.Uf = np.linalg.svd(Prs_perp @ self.target_cov @ Prs_perp)[0][:, :self.r0 - self.rs]
            self.finetuned_U0 = np.hstack((self.Urs, self.Uf))  


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

        self.delta_grid = np.asarray(np.logspace(-3, 0, num=10)) if delta_grid is None else np.asarray(delta_grid)
        self.tau_grid = np.asarray(np.logspace(-3, 0, num=10)) if tau_grid is None else np.asarray(tau_grid)
        
        self.selected_delta = None
        self.selected_tau = None

    def fit(self, max_iteration=10):
        """
        Cross-validation to select best delta and tau, then train TransPCA with selected parameters.
        """
        best_metric = float('-inf')
        best_delta = None
        best_tau = None

        total_norm = np.linalg.norm(self.target, 'fro')**2

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for delta_val in self.delta_grid:
            for tau_val in self.tau_grid:
                sum_metric = 0  

                for train_idx, val_idx in kf.split(np.asarray(self.target)): 
                    X_train, X_val = self.target[train_idx], self.target[val_idx]

                    temp_model = TransPCA(X_train, [np.asarray(source) for source in self.source_list],
                                          cov=self.cov, first_step=self.first_step, r0=self.r0,
                                          rk_list=self.rk_list, delta=delta_val, tau=tau_val)

                    temp_model.fit(max_iteration=10)

                    P = np.asarray(temp_model.finetuned_U0) @ np.asarray(temp_model.finetuned_U0).T
                    proj_val = P @ np.asarray(X_val).T
                    sum_metric += np.linalg.norm(proj_val, 'fro')**2

                info_retained = float(sum_metric) / float(total_norm)

                if info_retained > best_metric:
                    best_metric = info_retained
                    best_delta = delta_val
                    best_tau = tau_val

        self.selected_delta = best_delta
        self.selected_tau = best_tau
        print(f"Selected delta: {float(best_delta):.4f}, Selected tau: {float(best_tau):.4f}, Best information retention: {float(best_metric):.4f}")

        self.delta = self.selected_delta
        self.tau = self.selected_tau
        super().fit(max_iteration=max_iteration)