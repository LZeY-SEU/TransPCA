# 改进的MLP定义（带Kaiming初始化）
from time import time
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MLP(nn.Module):
    def __init__(self, p, d1, activation="ReLU"):
        super().__init__()
        self.fc1 = nn.Linear(p, d1, bias=False)
        self.dropout = nn.Dropout(0.5)  # Add Dropout layer

        self.fc2 = nn.Linear(d1, 1, bias=False)
        
        # Kaiming initialization (optimized for ReLU)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
        # Activation function selection
        self.activation = getattr(nn, activation)() if hasattr(nn, activation) else nn.Identity()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


# Training function (with full monitoring)
def train_mlp(X_train, y_train, p, d1=None, activation="ReLU", lr=0.005, epochs=50000):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data conversion (ensure memory continuity)
    X_torch = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_torch = torch.as_tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    
    # Model setup
    d1 = p if d1 is None else d1
    model = MLP(p, d1, activation).to(device)
    
    # Optimizer configuration (using AdamW for more stability)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    # Training loop
    start_time = time()
    for epoch in range(1, epochs + 1):
        # Forward pass
        y_pred = model(X_torch)
        loss = loss_fn(y_pred, y_torch)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing
        loss.backward()
        
        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Parameter update
        optimizer.step()
    
    
    # Final report
    #print(f"\nTraining completed in {time()-start_time:.2f} seconds")
    #print(f"Final loss: {loss.item():.6f}")
    
    return model



# Compute AGOP
def compute_gradient_W(model, X):
    """Calculate gradients and weight matrix, with full handling of device and gradient issues"""
    # Ensure model and data are on the same device
    device = next(model.parameters()).device
    
    # Convert data to model's device and enable gradients
    X_torch = torch.as_tensor(X, dtype=torch.float32).to(device)
    X_torch.requires_grad_(True)
    
    # Forward pass
    output = model(X_torch)
    
    # Compute gradients (retain computation graph)
    grads = torch.autograd.grad(
        outputs=output,
        inputs=X_torch,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]
    
    # Get weight matrix (automatically detached)
    W = model.fc1.weight
    
    # Safely convert to numpy arrays
    return grads.detach().cpu().numpy(), W.detach().cpu().numpy()


def get_models(nn_layer_size=30):
    models = {
        "Ridge Regression": Ridge(alpha=0.01),
        "Kernel Ridge (Exponential)": KernelRidge(kernel='laplacian', gamma=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Support Vector Regression": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(nn_layer_size,), max_iter=1000, random_state=42)
    }
    return models

