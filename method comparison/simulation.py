# use cvTransPCA to run 3 scenarios based on r_k^s = r_0^s
import torch
import random
import time
import numpy as np
import pandas as pd
import math
import gc
from scipy.stats import ortho_group
from transPCA import *

# Demetric
def Dmetric(PA1,PA2,p0): #compare two projection matrices with parameter (p,p0)
    return np.sqrt(1-np.trace(np.dot(PA1,PA2))/p0)

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# blind GB
def GB(Pt_list,nk_list,rs): 
    K = len(Pt_list)
    EPA = nk_list[0]*Pt_list[0].copy()
    for k in range(1,K):
        EPA += nk_list[k]*Pt_list[k].copy()
    UG = np.linalg.svd(EPA/np.sum(nk_list))[0][:,:rs]
    Ps = UG@UG.T
    return Ps 

def finetune(Sigma,Ps,rp,p):
    Ps_perp = np.eye(p)-Ps
    Uf = np.linalg.svd(Ps_perp@Sigma@Ps_perp)[0][:,:rp]
    return Uf@Uf.T + Ps

def GrassTPCA(target, source_list, K, p, rp, rs):
    Pks_list = []
    n_list = []
    r = rp + rs
    n_list.append(len(target))
    x_0 = target
    Sigma_0 = x_0.T@x_0/(len(target)-1)
    U0 = np.linalg.svd(Sigma_0)[0][:,:r]
    Pks_list.append(U0@U0.T)
    for k in range(K-1):
        n_list.append(len(source_list[k]))
        x_k = source_list[k]
        Sigma_k = x_k.T@x_k/(len(source_list[k])-1)
        Uk = np.linalg.svd(Sigma_k)[0][:,:r]
        Pks_list.append(Uk@Uk.T)
    Ps = GB(Pks_list,n_list,rs)
    P0s_finetuned = finetune(Sigma_0,Ps,rp,p)
    return P0s_finetuned


# blind PCA
def poolPCA(target, source_list, p, n, K, rp, rs):
    r = rs + rp
    X = np.zeros((p,n*K))
    X[:,0:n] = np.array(target.T)
    for k in range(1,K):
        X[:,(n*k):(n+n*k)] = np.array(source_list[k-1].T)
    Sigma = X@X.T/(n*K)
    U = np.linalg.svd(Sigma)[0][:,:r]
    return U@U.T

def generate_simulation_data_1(n, p, h, rs, rp, K, fs, fp):
    """
    Generate simulated data for transPCA experiments.

    Parameters:
    - n (int): Number of samples per dataset.
    - p (int): Feature dimension.
    - h (float): Deviation factor between target and source datasets.
    - rs (int): Shared subspace dimension.
    - rp (int): Private subspace dimension per source dataset.
    - K (int): Total number of datasets (including target).
    - fs (int): Eigenvalues for shared subspace
    - fp (int): Eigenvalues for private subspace

    Returns:
    - target (ndarray): Target dataset of shape (n, p).
    - source_list (list of ndarray): List of K-1 source datasets, each of shape (n, p).
    - P0 (ndarray): True projection matrix for target (p, rs).
    """

    # Generate base shared and private subspaces
    U = ortho_group.rvs(dim=p)
    Uc = U[:,:rs]  
    v = h/np.sqrt(p)
    Ip = np.eye(p)

    # Generate target subspaces
    E0 = v * np.random.normal(0,1,(p,p))
    U0s = np.linalg.svd(Uc@Uc.T + E0)[0][:,:rs]
    U0p = np.linalg.svd(Ip - U0s@U0s.T)[0][:,:rp]
    P0 = U0p@U0p.T + U0s@U0s.T
    U0_orth = np.linalg.svd(Ip - P0)[0][:,:(p-rp-rs)] 

    # Generate Gaussian target dataset
    Lambda = np.diag(np.ones(p))
    for i in range(rp):
        Lambda[i,i] = fp
    for i in range(rp,(rp + rs)):
        Lambda[i,i] = fs
    Sigma_target = np.dot(np.block([U0p,U0s,U0_orth]),
                   np.dot(Lambda,np.block([U0p,U0s,U0_orth]).T))  # Target covariance matrix
    target = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_target, size=n)

    # Generate source datasets
    source_list = []
    for k in range(1,K):
        # Generate source subspaces
        Ek = v * np.random.normal(0,1,(p,p))
        Uks = np.linalg.svd(Uc@Uc.T + Ek)[0][:,:rs]
        Ukp = np.linalg.svd(Ip - Uks@Uks.T)[0][:,:rp]
        Pk = Ukp@Ukp.T + Uks@Uks.T
        Uk_orth = np.linalg.svd(Ip - Pk)[0][:,:(p-rp-rs)] 

        # Generate covariance and dataset
        Sigma_source = np.dot(np.block([Ukp,Uks,Uk_orth]),
                   np.dot(Lambda,np.block([Ukp,Uks,Uk_orth]).T))  # Target covariance matrix
        source_data = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_source, size=n)
        source_list.append(source_data)

    return target, source_list, P0

def generate_simulation_data_2(n, p, h, rs, rp, K, K_useless, fs, fp):
    """
    Generate simulated data for transPCA experiments.

    Parameters:
    - n (int): Number of samples per dataset.
    - p (int): Feature dimension.
    - h (float): Deviation factor between target and source datasets.
    - rs (int): Shared subspace dimension.
    - rp (int): Private subspace dimension per source dataset.
    - K (int): Total number of datasets (including target).
    - K_useless (int): number of useless datasets
    - fs (int): Eigenvalues for shared subspace
    - fp (int): Eigenvalues for private subspace

    Returns:
    - target (ndarray): Target dataset of shape (n, p).
    - source_list (list of ndarray): List of K-1 source datasets, each of shape (n, p).
    - P0 (ndarray): True projection matrix for target (p, rs).
    """

    # Generate base shared and private subspaces
    U = ortho_group.rvs(dim=p)
    Uc = U[:,:rs]  
    v = h/np.sqrt(p)
    Ip = np.eye(p)

    # Generate target subspaces
    E0 = v * np.random.normal(0,1,(p,p))
    U0s = np.linalg.svd(Uc@Uc.T + E0)[0][:,:rs]
    U0p = np.linalg.svd(Ip - U0s@U0s.T)[0][:,:rp]
    P0 = U0p@U0p.T + U0s@U0s.T
    U0_orth = np.linalg.svd(Ip - P0)[0][:,:(p-rp-rs)] 

    # Generate Gaussian target dataset
    Lambda = np.diag(np.ones(p))
    for i in range(rp):
        Lambda[i,i] = fp
    for i in range(rp,(rp + rs)):
        Lambda[i,i] = fs
    Sigma_target = np.dot(np.block([U0p,U0s,U0_orth]),
                   np.dot(Lambda,np.block([U0p,U0s,U0_orth]).T))  # Target covariance matrix
    target = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_target, size=n)

    # Generate source datasets
    source_list = []
    for k in range(1,(K - K_useless)):
        # Generate source subspaces
        Ek = v * np.random.normal(0,1,(p,p))
        Uks = np.linalg.svd(Uc@Uc.T + Ek)[0][:,:rs]
        Ukp = np.linalg.svd(Ip - Uks@Uks.T)[0][:,:rp]
        Pk = Ukp@Ukp.T + Uks@Uks.T
        Uk_orth = np.linalg.svd(Ip - Pk)[0][:,:(p-rp-rs)] 

        # Generate covariance and dataset
        Sigma_source = np.dot(np.block([Ukp,Uks,Uk_orth]),
                   np.dot(Lambda,np.block([Ukp,Uks,Uk_orth]).T))  # Target covariance matrix
        source_data = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_source, size=n)
        source_list.append(source_data)

    for k in range((K - K_useless),K):
        U_useless = ortho_group.rvs(dim=p)
        Sigma_source = np.dot(U_useless,np.dot(Lambda,U_useless.T))
        source_data = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_source, size=n)
        source_list.append(source_data)

    return target, source_list, P0

def generate_simulation_data_3(n, p, h, rs, rp, K, K_useless, fs, fp):
    """
    Generate simulated data for transPCA experiments.

    Parameters:
    - n (int): Number of samples per dataset.
    - p (int): Feature dimension.
    - h (float): Deviation factor between target and source datasets.
    - rs (int): Shared subspace dimension.
    - rp (int): Private subspace dimension per source dataset.
    - K (int): Total number of datasets (including target).
    - K_useless (int): number of useless datasets
    - fs (int): Eigenvalues for shared subspace
    - fp (int): Eigenvalues for private subspace

    Returns:
    - target (ndarray): Target dataset of shape (n, p).
    - source_list (list of ndarray): List of K-1 source datasets, each of shape (n, p).
    - P0 (ndarray): True projection matrix for target (p, rs).
    """

    # Generate base shared and private subspaces
    U = ortho_group.rvs(dim=p)
    Uc = U[:,:rs]  
    V = ortho_group.rvs(dim=p)
    Vc = V[:, :rs]
    v = h/np.sqrt(p)
    Ip = np.eye(p)

    # Generate target subspaces
    E0 = v * np.random.normal(0,1,(p,p))
    U0s = np.linalg.svd(Uc@Uc.T + E0)[0][:,:rs]
    U0p = np.linalg.svd(Ip - U0s@U0s.T)[0][:,:rp]
    P0 = U0p@U0p.T + U0s@U0s.T
    U0_orth = np.linalg.svd(Ip - P0)[0][:,:(p-rp-rs)] 

    # Generate Gaussian target dataset
    Lambda = np.diag(np.ones(p))
    for i in range(rp):
        Lambda[i,i] = fp
    for i in range(rp,(rp + rs)):
        Lambda[i,i] = fs
    Sigma_target = np.dot(np.block([U0p,U0s,U0_orth]),
                   np.dot(Lambda,np.block([U0p,U0s,U0_orth]).T))  # Target covariance matrix
    target = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_target, size=n)

    # Generate source datasets
    source_list = []
    for k in range(1,(K - K_useless)):
        # Generate source subspaces
        Ek = v * np.random.normal(0,1,(p,p))
        Uks = np.linalg.svd(Uc@Uc.T + Ek)[0][:,:rs]
        Ukp = np.linalg.svd(Ip - Uks@Uks.T)[0][:,:rp]
        Pk = Ukp@Ukp.T + Uks@Uks.T
        Uk_orth = np.linalg.svd(Ip - Pk)[0][:,:(p-rp-rs)] 

        # Generate covariance and dataset
        Sigma_source = np.dot(np.block([Ukp,Uks,Uk_orth]),
                   np.dot(Lambda,np.block([Ukp,Uks,Uk_orth]).T))  # Target covariance matrix
        source_data = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_source, size=n)
        source_list.append(source_data)

    for k in range((K - K_useless),K):
        # Generate source subspaces
        Ek = v * np.random.normal(0,1,(p,p))
        Uks = np.linalg.svd(Vc@Vc.T + Ek)[0][:,:rs]
        Ukp = np.linalg.svd(Ip - Uks@Uks.T)[0][:,:rp]
        Pk = Ukp@Ukp.T + Uks@Uks.T
        Uk_orth = np.linalg.svd(Ip - Pk)[0][:,:(p-rp-rs)] 

        # Generate covariance and dataset
        Sigma_source = np.dot(np.block([Ukp,Uks,Uk_orth]),
                   np.dot(Lambda,np.block([Ukp,Uks,Uk_orth]).T))  # Target covariance matrix
        source_data = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma_source, size=n)
        source_list.append(source_data)

    return target, source_list, P0

def run_simulation(n=100, p=50, h=0.1, rs=6, rp=3, fs=10, fp=20, n_folds=5, iterate=100):
    """
    Run cvTransPCA on generated data and evaluate its performance.

    Parameters:
    - n (int): Number of samples per dataset.
    - p (int): Feature dimension.
    - h (float): Deviation factor between target and source datasets.
    - rs (int): Shared subspace dimension.
    - rp (int): Private subspace dimension per source dataset.
    - K (int): Total number of datasets (including target).
    - K_useless (int): number of useless datasets
    - fs (int): Eigenvalues for shared subspace
    - fp (int): Eigenvalues for private subspace
    - n_folds (int): Number of folds in cross-validation.
    """

    set_random_seed(0)
    
    # Start timing
    start_time = time.time()

    
    # Generate data
    r = rs + rp
    K_list = [7,13,19,25]
    Error_average = np.zeros((4,len(K_list),3)) # 4 methods with varying K under 3 scenarios
    for i in range(len(K_list)):
        K = K_list[i]
        K_useless = int((K-1)/3)
        for t in range(iterate):
            target_1, source_list_1, P0_1 = generate_simulation_data_1(n, p, h, rs, rp, K, fs, fp)
            target_2, source_list_2, P0_2 = generate_simulation_data_2(n, p, h, rs, rp, K, K_useless, fs, fp)
            target_3, source_list_3, P0_3 = generate_simulation_data_3(n, p, h, rs, rp, K, K_useless, fs, fp)
            
            # Run different methods under 3 backgrounds
            # S.1
            # cvTransPCA
            model_1 = cvTransPCA(target_1, source_list_1, cov='covariance', n_folds=n_folds)
            model_1.fit()  # This will now handle both parameter selection and model fitting
            # BlindGB
            P0_blindGB_1 = GrassTPCA(target_1,source_list_1,K,p,rp,rs)
            # BlindPCA
            P0_blindPCA_1 = poolPCA(target_1,source_list_1,p,n,K,rp,rs)
        
            # S.2
            # cvTransPCA
            model_2 = cvTransPCA(target_2, source_list_2, cov='covariance', n_folds=n_folds)
            model_2.fit()
            # BlindGB
            P0_blindGB_2 = GrassTPCA(target_2,source_list_2,K,p,rp,rs)
            # BlindPCA
            P0_blindPCA_2 = poolPCA(target_2,source_list_2,p,n,K,rp,rs)
        
            # S.3
            # cvTransPCA
            model_3 = cvTransPCA(target_3, source_list_3, cov='covariance', n_folds=n_folds)
            model_3.fit()
            # BlindGB
            P0_blindGB_3 = GrassTPCA(target_3,source_list_3,K,p,rp,rs)
            # BlindPCA
            P0_blindPCA_3 = poolPCA(target_3,source_list_3,p,n,K,rp,rs)
        
    
        
            # Compute Frobenius norm errors
            Error_average[0,i,0] += Dmetric(model_1.U0 @ model_1.U0.T, P0_1, r)
            Error_average[1,i,0] += Dmetric(model_1.finetuned_U0 @ model_1.finetuned_U0.T, P0_1, r)
            Error_average[2,i,0] += Dmetric(P0_blindGB_1, P0_1, r)
            Error_average[3,i,0] += Dmetric(P0_blindPCA_1, P0_1, r)
        
        
            Error_average[0,i,1] += Dmetric(model_2.U0 @ model_2.U0.T, P0_2, r)
            Error_average[1,i,1] += Dmetric(model_2.finetuned_U0 @ model_2.finetuned_U0.T, P0_2, r)
            Error_average[2,i,1] += Dmetric(P0_blindGB_2, P0_2, r)
            Error_average[3,i,1] += Dmetric(P0_blindPCA_2, P0_2, r)
        
        
            Error_average[0,i,2] += Dmetric(model_3.U0 @ model_3.U0.T, P0_3, r)
            Error_average[1,i,2] += Dmetric(model_3.finetuned_U0 @ model_3.finetuned_U0.T, P0_3, r)
            Error_average[2,i,2] += Dmetric(P0_blindGB_3, P0_3, r)
            Error_average[3,i,2] += Dmetric(P0_blindPCA_3, P0_3, r)

            del target_1, source_list_1, P0_1, model_1, P0_blindGB_1, P0_blindPCA_1
            del target_2, source_list_2, P0_2, model_2, P0_blindGB_2, P0_blindPCA_2
            del target_3, source_list_3, P0_3, model_3, P0_blindGB_3, P0_blindPCA_3
            gc.collect()

    
    row_labels = ["individual PCA", "selected GB", "blind GB", "blind PCA"]

    column_labels = [f"K={K_list[i]}" for i in range(len(K_list))]

    num_third_dim = Error_average.shape[2]

    for k in range(num_third_dim):
        print(f"Average Error (Scenario {k+1}):")
        print("            " + "   ".join(column_labels))  
        for i, row in enumerate(Error_average[:, :, k]/iterate):
            print(f"{row_labels[i]:<15} " + "  ".join(f"{x:.4f}" for x in row))
        print()  
    
    
    
    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Total runtime: {total_time_minutes:.2f} minutes")