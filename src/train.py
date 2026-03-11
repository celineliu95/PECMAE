import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
from sklearn.metrics import balanced_accuracy_score

from model import PrototypicalNetwork
from utils_dataset import create_dataloaders

def init_prototypes_with_kmeans(model, dataloader, device):
    
    model.eval()
    all_zx = []
    all_labels = []
    
    with torch.no_grad():
        for z_x, labels in dataloader:
            all_zx.append(z_x.cpu())
            all_labels.append(labels.cpu())
            
    all_zx = torch.cat(all_zx, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    new_prototypes = torch.zeros_like(model.prototypes.data).cpu()

    for c in range(model.num_classes):
        
        class_zx = all_zx[all_labels == c]
        
        if len(class_zx) < model.num_prototypes_per_class:
            raise ValueError(f"Warning: class {c} has only {len(class_zx)} samples, which is less than the number of prototypes per class ({model.num_prototypes_per_class}).")
            
        kmeans = KMeans(n_clusters=model.num_prototypes_per_class, n_init=10, random_state=42)
        kmeans.fit(class_zx)
        
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        start_idx = c * model.num_prototypes_per_class
        end_idx = start_idx + model.num_prototypes_per_class
        new_prototypes[start_idx:end_idx] = centroids
        
    model.prototypes.data.copy_(new_prototypes.to(device))  
  
def train_step(model, z_x, labels, optimizer, scheduler, lambda_weight, device):
    
    model.train()
    z_x, labels = z_x.to(device), labels.to(device)
    
    optimizer.zero_grad()
    
    logits, S, z_p = model(z_x)
    
    loss, loss_c, loss_p = model.compute_loss(logits, z_x, z_p, labels, lambda_weight)
    
    loss.backward()
    
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
        
    return loss.item(), loss_c.item(), loss_p.item()


def train_model(model, train_loader, num_epochs, lambda_weight=0.25, device='cuda'):

    model = model.to(device)
    
    init_prototypes_with_kmeans(model, train_loader, device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_steps
    )
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_loss_c = 0.0
        epoch_loss_p = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for z_x, labels in pbar:
            
            loss, loss_c, loss_p = train_step(
                model=model, 
                z_x=z_x, 
                labels=labels, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                lambda_weight=lambda_weight, 
                device=device
            )
            
            epoch_loss += loss
            epoch_loss_c += loss_c
            epoch_loss_p += loss_p
            
            pbar.set_postfix({
                'Total': f"{loss:.4f}", 
                'Lc': f"{loss_c:.4f}", 
                'Lp': f"{loss_p:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
        avg_loss = epoch_loss / len(train_loader)
        avg_lc = epoch_loss_c / len(train_loader)
        avg_lp = epoch_loss_p / len(train_loader)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Lc: {avg_lc:.4f} | Lp: {avg_lp:.4f}")

    print("\nDone.")
    return model

def test_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for z_x, labels in test_loader:
            z_x = z_x.to(device)
            
            logits, _, _ = model(z_x)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    normalized_acc = balanced_accuracy_score(all_targets, all_preds)
    
    print(f"Balanced Accuracy : {normalized_acc * 100:.2f}%")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lambda_weight', type=float, default=0.25)
    parser.add_argument('--num_prototypes_per_class', type=int, default=5)
    parser.add_argument('--use_adaptor', type=bool, default=True)
    parser.add_argument('--features_file', type=str, default='data/features.pt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = PrototypicalNetwork(num_classes=8, num_prototypes_per_class=args.num_prototypes_per_class, embedding_dim=768, use_adaptor=args.use_adaptor)

    train_loader, test_loader = create_dataloaders(
        features_file=args.features_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        lambda_weight=args.lambda_weight,
        device=device
    )
    
    torch.save(trained_model.state_dict(), 'saves/prototypical_network.pth')
    
    test_model(trained_model, test_loader, device=device)