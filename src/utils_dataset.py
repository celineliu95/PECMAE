import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloaders(features_file, batch_size=256, train_ratio=0.8, num_workers=4, seed=42):
    
    set_seed(seed)
    
    data = torch.load(features_file)
    X = data['X'] # shape (N, 768)
    y = data['y'] # shape (N,)
    
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, 
        train_size=train_ratio, 
        stratify=y.numpy(),
        random_state=seed
    )
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True)
    std[std == 0] = 1e-6
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"Created loaders -> Train: {len(train_dataset)} samples | Test: {len(test_dataset)} samples.")
    return train_loader, test_loader