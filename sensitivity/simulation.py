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
