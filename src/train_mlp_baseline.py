import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import balanced_accuracy_score

from utils_dataset import create_dataloaders

class SimpleMLP(nn.Module):
    def __init__(self, in_size=768, hidden_size=256, out_size=8):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )
        
    def forward(self, x):
        return self.net(x)

def eval_model(model, loader, criterion, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []

    total_loss = 0.0
    
    with torch.no_grad():
        for z_x, labels in loader:
            z_x, labels = z_x.to(device), labels.to(device)
            
            logits = model(z_x)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    normalized_acc = balanced_accuracy_score(all_targets, all_preds)
    
    print(f"Eval | Loss: {total_loss / len(loader):.4f} | Balanced Accuracy : {normalized_acc * 100:.2f}%")
    
    return total_loss / len(loader)

def train_step(model, z_x, labels, optimizer, criterion, device):
    model.train()
    z_x, labels = z_x.to(device), labels.to(device)
    
    optimizer.zero_grad()

    logits = model(z_x)

    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_mlp(model, train_loader, val_loader, num_epochs, device='cuda', save_path='saves/best_mlp_baseline.pth'):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for z_x, labels in pbar:

            loss = train_step(model, z_x, labels, optimizer, criterion, device)
            epoch_loss += loss
            pbar.set_postfix({'Loss': f"{loss:.4f}"})
            
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
        
        val_loss = eval_model(model, val_loader, criterion, device=device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        
    print("\nTraining complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline MLP on Medley features")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--features_file', type=str, default='data/medley_features.pt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='saves/best_mlp_baseline.pth')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, val_loader, test_loader, _, _ = create_dataloaders(
        features_file=args.features_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    model = SimpleMLP(in_size=768, hidden_size=args.hidden_size, out_size=8)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Baseline MLP Training (Hidden Size: {args.hidden_size}) ---")
    
    train_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        device=device,
        save_path=args.save_path
    )
    
    print("\n--- Final Evaluation on Test Set ---")
    best_model = SimpleMLP(in_size=768, hidden_size=args.hidden_size, out_size=8).to(device)
    best_model.load_state_dict(torch.load(args.save_path, map_location=device))
    eval_model(best_model, test_loader, criterion, device=device)