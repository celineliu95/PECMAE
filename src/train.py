import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
from sklearn.metrics import balanced_accuracy_score
import warnings

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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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

def extract_latent_states(model, dataloader, device):
    model.eval()
    all_zx = []
    all_labels = []
    with torch.no_grad():
        for z_x, labels in dataloader:
            all_zx.append(z_x.cpu())
            all_labels.append(labels.cpu())
            
    zx_tensor = torch.cat(all_zx, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    zp_tensor = model.get_projected_prototypes().cpu().detach()
    
    return zx_tensor, labels_tensor, zp_tensor

def eval_model(model, loader, lambda_weight, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    
    total_loss = 0.0
    
    with torch.no_grad():
        for z_x, labels in loader:
            z_x = z_x.to(device)
            
            logits, S, z_p = model(z_x)
            
            loss, _, _ = model.compute_loss(logits, z_x, z_p, labels, lambda_weight)
            total_loss += loss
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    normalized_acc = balanced_accuracy_score(all_targets, all_preds)
    
    print(f"Eval | Loss: {total_loss / len(loader):.4f} | Balanced Accuracy : {normalized_acc * 100:.2f}%")
    
    return total_loss / len(loader)

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

def train_model(model, train_loader, num_epochs, lambda_weight=0.25, device='cuda', save_path='saves/best_prototypical_network.pth'):

    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_steps
    )
    #scheduler = None
    
    best_val_loss = float("inf")
    
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
            
            if scheduler is None:
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Total': f"{loss:.4f}", 
                'Lc': f"{loss_c:.4f}", 
                'Lp': f"{loss_p:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
        avg_loss = epoch_loss / len(train_loader)
        avg_lc = epoch_loss_c / len(train_loader)
        avg_lp = epoch_loss_p / len(train_loader)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Lc: {avg_lc:.4f} | Lp: {avg_lp:.4f}")
        
        val_loss = eval_model(model, val_loader, args.lambda_weight, device=device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        
    print("\nTraining done.")
    return model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lambda_weight', type=float, default=0.25)
    parser.add_argument('--num_prototypes_per_class', type=int, default=5)
    parser.add_argument('--use_adaptor', type=bool, default=True)
    parser.add_argument('--features_file', type=str, default='data/features.pt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_linear', action='store_true', help="Freeze the linear classification layer")
    parser.add_argument('--save_path', type=str, default='saves/best_prototypical_network.pth')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = PrototypicalNetwork(num_classes=8, num_prototypes_per_class=args.num_prototypes_per_class, embedding_dim=768, use_adaptor=args.use_adaptor, freeze_linear=args.freeze_linear)
    
    train_loader, val_loader, test_loader, train_mean, train_std = create_dataloaders(
        features_file=args.features_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    model.set_normalization_stats(train_mean.to(device), train_std.to(device))
    
    print("\nEvaluating untrained model on test set...")
    eval_model(model, test_loader, args.lambda_weight, device=device)

    print("\nInitializing prototypes with K-Means")
    init_prototypes_with_kmeans(model, train_loader, device)

    print("\nEvaluating Zero-Shot K-Means")
    eval_model(model, test_loader, args.lambda_weight, device=device)

    zx_test, labels_test, zp_before = extract_latent_states(model, test_loader, device)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        lambda_weight=args.lambda_weight,
        device=device,
        save_path=args.save_path
    )
    
    print("\nEvaluating trained model on test set...")
    eval_model(trained_model, test_loader, args.lambda_weight, device=device)
    
    best_model = PrototypicalNetwork(num_classes=8, num_prototypes_per_class=args.num_prototypes_per_class, embedding_dim=768, use_adaptor=args.use_adaptor).to(device)
    best_model.load_state_dict(torch.load("saves/best_prototypical_network.pth", map_location=device))
    
    print("\nEvaluating best model on test set...")
    eval_model(best_model, test_loader, args.lambda_weight, device=device)

    _, _, zp_after = extract_latent_states(best_model, test_loader, device)
    latent_data_path = args.save_path.replace('.pth', '_latent_tensors.pt')
    torch.save({
        'zx': zx_test,            # Real samples embeddings
        'labels': labels_test,    # Corresponding class labels
        'zp_before': zp_before,   # Prototypes at K-Means init
        'zp_after': zp_after      # Prototypes after final training
    }, latent_data_path)