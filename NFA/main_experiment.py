import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from time import time
import cupy as cp
from sklearn.model_selection import train_test_split


from algorithms import *
from simulation import *
from TransPCA_GPU import * 

def compare_learning_strategies(models, 
                              X_train_target, y_train_target,
                              X_test_target, y_test_target,
                              pooled_X, pooled_y,
                              U0_direct, U0_transfer):
    """
    Compare model performance across 4 learning strategies:
    1. Directly on target dataset
    2. On pooled dataset (all sources + target)
    3. Using dimensionality reduction with U0 from target
    4. Using dimensionality reduction with U0 from transfer learning
    
    Args:
        models: Dictionary of model objects
        X_train_target/y_train_target: Target training data
        X_test_target/y_test_target: Target test data
        pooled_X/pooled_y: Combined data from all sources + target
        U0_direct: Subspace from target data (shape [p, r])
        U0_transfer: Subspace from transfer learning (shape [p, r])
    
    Returns:
        Tuple of 4 DataFrames with evaluation results for each strategy
    """
    # Strategy 1: Direct target training
    df_target = evaluate_models(models, X_train_target, y_train_target, 
                              X_test_target, y_test_target)
    df_target['Strategy'] = 'Target Only'
    
    # Strategy 2: Pooled data training
    df_pooled = evaluate_models(models, pooled_X, pooled_y,
                              X_test_target, y_test_target)
    df_pooled['Strategy'] = 'Pooled Data'
    
    # Strategy 3: Target U0 projection
    X_train_proj = X_train_target @ U0_direct
    X_test_proj = X_test_target @ U0_direct
    df_proj_direct = evaluate_models(models, X_train_proj, y_train_target,
                                   X_test_proj, y_test_target)
    df_proj_direct['Strategy'] = 'Target Subspace'
    
    # Strategy 4: Transfer U0 projection
    X_train_transfer = X_train_target @ U0_transfer
    X_test_transfer = X_test_target @ U0_transfer
    df_proj_transfer = evaluate_models(models, X_train_transfer, y_train_target,
                                     X_test_transfer, y_test_target)
    df_proj_transfer['Strategy'] = 'Transfer Subspace'
    
    return df_target, df_pooled, df_proj_direct, df_proj_transfer

def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Evaluate multiple models on given datasets
    
    Args:
        models: Dictionary {model_name: model_instance}
        X_train/y_train: Training data
        X_test/y_test: Test data
    
    Returns:
        DataFrame with evaluation metrics
    """
    results = []
    
    for name, model in models.items():
        start_time = time()
        
        try:
            model.fit(X_train, y_train.ravel())
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            results.append({
                'Model': name,
                'Train R²': r2_score(y_train, train_pred),
                'Test R²': r2_score(y_test, test_pred),
                'Time (s)': time() - start_time
            })
            
        except Exception as e:
            print(f"{name} failed: {str(e)}")
            results.append({
                'Model': name,
                'Train R²': np.nan,
                'Test R²': np.nan,
                'Time (s)': np.nan
            })
    
    return pd.DataFrame(results)

def generate_comparison_report(df_target, df_pooled, df_proj_direct, df_proj_transfer):
    """
    Generate a consolidated report comparing all strategies
    
    Args:
        DataFrames from compare_learning_strategies()
    
    Returns:
        Consolidated DataFrame with multi-index columns
    """
    # Add strategy prefix to columns
    dfs = []
    for df, strategy in zip([df_target, df_pooled, df_proj_direct, df_proj_transfer],
                          ['Target', 'Pooled', 'DirectU0', 'TransferU0']):
        df = df.set_index('Model')
        df.columns = [f"{strategy}_{col}" for col in df.columns]
        dfs.append(df)
    
    # Combine all results
    full_report = pd.concat(dfs, axis=1)
    
    # Calculate improvement metrics
    full_report['Improvement_Pooled'] = full_report['Pooled_Test R²'] - full_report['Target_Test R²']
    full_report['Improvement_Transfer'] = full_report['TransferU0_Test R²'] - full_report['DirectU0_Test R²']
    
    return full_report.sort_values(by='Target_Test R²', ascending=False)




def run_experiment(K, n, p, rs, rp, g_types, replications=10, test_size=0.2, random_state=42):
    """
    Main experiment comparing different learning strategies across multiple replications
    
    Args:
        K: Number of datasets
        n: Samples per dataset
        p: Feature dimension
        rs: Shared subspace dimension
        rp: Private subspace dimension
        g_types: List of nonlinear function types for each dataset
        replications: Number of experimental replications
        test_size: Proportion for test split
        random_state: Random seed
    
    Returns:
        tuple: (strategy_comparison_df, subspace_distance_df)
    """
    # Initialize result containers
    strategy_results = []
    subspace_results = []
    
    for rep in range(replications):
        # Generate synthetic data
        current_seed = random_state + rep
        datasets, (Us, Up0) = generate_multisource_data(K, n, p, rs, rp, g_types, random_state=current_seed)
        
        # Prepare target dataset
        X_target, y_target = datasets[0]
        X_train_target, X_test_target, y_train_target, y_test_target = train_test_split(
            X_target, y_target, test_size=test_size, random_state=current_seed)
        
        # Prepare source datasets
        train_X_list = [X_train_target] + [datasets[_][0] for _ in range(1, K)]
        train_y_list = [y_train_target] + [datasets[_][1] for _ in range(1, K)]
        
        # Compute gradients and weights
        grad_list = []
        W_list = []
        for i in range(len(train_X_list)):
            trained_model = train_mlp(
                train_X_list[i], 
                train_y_list[i],
                p=p,
                d1=p,
                activation="ReLU",
                lr=0.001,
                epochs=5000
            )
            grad, W = compute_gradient_W(trained_model, train_X_list[i])
            grad_list.append(grad)
            W_list.append(W)
        
        # Perform subspace analysis
        target = grad_list[0]
        source_list = grad_list[1:]
        model_TPCA = TransPCA(
            target, source_list,
            first_step='GB',
            cov='covariance',
            tau=1,
            delta=0.1,
            r0=rs+rp,
            rk_list=[rs+rp]*(K-1),
            r0s = rs
        )
        model_TPCA.fit()
        
        # Compute true subspace (U = [Us, Up0])
        U = np.hstack([Us, Up0])
        r = rs + rp  # Total subspace dimension
        
        # Calculate subspace distances
        direct_distance = np.linalg.norm(
            cp.asnumpy(model_TPCA.U0) @ cp.asnumpy(model_TPCA.U0).T - U @ U.T, 'fro') / np.sqrt(2 * r)
        transfer_distance = np.linalg.norm(
            cp.asnumpy(model_TPCA.finetuned_U0) @ cp.asnumpy(model_TPCA.finetuned_U0).T - U @ U.T, 'fro') / np.sqrt(2 * r)
        
        # Record subspace results
        subspace_results.append({
            'Replication': rep,
            'Direct_Subspace_Distance': direct_distance,
            'Transfer_Subspace_Distance': transfer_distance,
            'Seed': current_seed
        })
        
        # Prepare pooled data
        X_pooled = np.vstack(train_X_list)
        y_pooled = np.concatenate(train_y_list)
        
        # Compare learning strategies
        df_target, df_pooled, df_direct, df_transfer = compare_learning_strategies(
            models=get_models(),
            X_train_target=X_train_target,
            y_train_target=y_train_target,
            X_test_target=X_test_target,
            y_test_target=y_test_target,
            pooled_X=X_pooled,
            pooled_y=y_pooled,
            U0_direct=cp.asnumpy(model_TPCA.U0),
            U0_transfer=cp.asnumpy(model_TPCA.finetuned_U0)
        )
        
        # Add replication info to strategy results
        for df in [df_target, df_pooled, df_direct, df_transfer]:
            df['Replication'] = rep
            df['Seed'] = current_seed
        strategy_results.extend([df_target, df_pooled, df_direct, df_transfer])
    
    # Combine all results
    strategy_comparison_df = pd.concat(strategy_results, ignore_index=True)
    subspace_distance_df = pd.DataFrame(subspace_results)
    
    return strategy_comparison_df, subspace_distance_df

