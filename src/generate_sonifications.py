import sys

import torch
from collections import defaultdict

_old_to = torch.Tensor.to
def patched_to(self, *args, **kwargs):
    if any(isinstance(a, str) and 'cuda' in a for a in args) or \
       kwargs.get('device', '') == 'cuda':
        return _old_to(self, 'cpu')
    return _old_to(self, *args, **kwargs)

torch.Tensor.to = patched_to
_old_module_to = torch.nn.Module.to
def patched_module_to(self, *args, **kwargs):
    return _old_module_to(self, 'cpu')
torch.nn.Module.to = patched_module_to

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.is_initialized = lambda: False
torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import argparse
from scipy.io import wavfile
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils_dataset import create_dataloaders
from model import PrototypicalNetwork

import encodecmae_to_wav.hub as hub

_original_get_model = hub.get_model
def patched_get_model(*args, **kwargs):
    kwargs.pop('processor', None)
    return _original_get_model(*args, **kwargs)
hub.get_model = patched_get_model

MEDLEY_INSTRUMENTS = {
    0: "clarinet",
    1: "distorted_electric_guitar",
    2: "female_singer",
    3: "flute",
    4: "piano",
    5: "tenor_saxophone",
    6: "trumpet",
    7: "violin"
}

def compute_mean_diversity(prototypes_tensor, num_classes=8):
    n_zp = prototypes_tensor.shape[0]
    num_prototypes_per_class = n_zp // num_classes
    
    if num_prototypes_per_class <= 1:
        return None
        
    diversities = []
    for c in range(num_classes):
        start_idx = c * num_prototypes_per_class
        end_idx = start_idx + num_prototypes_per_class
        
        class_protos = prototypes_tensor[start_idx:end_idx]
        div = torch.pdist(class_protos, p=2).mean().item()
        diversities.append(div)
        
    return float(np.mean(diversities))

def generate_and_save_prototypes(trained_model, diff_model, num_prototypes, output_dir="prototypes_audio", device="cuda"):
    assert trained_model.num_prototypes_per_class >= num_prototypes, "num_prototypes must be less than or equal to the number of prototypes per class in the model."
    
    trained_model.eval()
    trained_model.to(device)
    diff_model.model.eval()
    diff_model.model.to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving prototypes to: prototypes_audio/{output_dir}/")
    
    with torch.no_grad():
        for class_idx in range(trained_model.num_classes):
            
            instr_name = MEDLEY_INSTRUMENTS.get(class_idx, f"class_{class_idx}")
            class_dir = os.path.join(output_dir, f"{class_idx}_{instr_name}")
            os.makedirs(class_dir, exist_ok=True)
            
            print(f"Generating prototypes for class: {instr_name}...")
            for p_idx in range(num_prototypes):
                
                z_p = trained_model.get_prototype(class_idx, p_idx, projected=True, denormalize=True)
                
                z_p_batch = z_p.to(device)
                
                waveform = diff_model.sample(z_p_batch)
                wave_int16 = np.int16(waveform * 32767)
                
                filename = f"prototype_{p_idx}.wav"
                filepath = os.path.join(class_dir, filename)
                
                wavfile.write(filepath, 24000, wave_int16)
                
    print("Prototypes generated.")

def generate_zeroshot_kmeans_audio(trained_model, diff_model, latent_file_path, num_prototypes_to_save, output_dir="medley_experiments", device="cpu"):
    if not os.path.exists(latent_file_path):
        print(f"Error: Latent file {latent_file_path} not found. Cannot generate zero-shot audio.")
        return
        
    print(f"\nLoading zero-shot K-Means prototypes from {latent_file_path}...")
    data = torch.load(latent_file_path, map_location="cpu")
    zp_before = data['zp_before'] 
    
    diff_model.model.eval()
    if hasattr(diff_model.model, 'to'):
        diff_model.model.to(device)
        
    zero_shot_dir = os.path.join(output_dir, "zeroshot_kmeans")
    os.makedirs(zero_shot_dir, exist_ok=True)
    print(f"Saving Zero-Shot K-Means prototypes to: {zero_shot_dir}/")

    mean = trained_model.data_mean.cpu()
    std = trained_model.data_std.cpu()

    with torch.no_grad():
        for class_idx in range(trained_model.num_classes):
            instr_name = MEDLEY_INSTRUMENTS.get(class_idx, f"class_{class_idx}")
            class_dir = os.path.join(zero_shot_dir, f"{class_idx}_{instr_name}")
            os.makedirs(class_dir, exist_ok=True)
            
            print(f"Generating Zero-Shot prototypes for class: {instr_name}...")
            for p_idx in range(num_prototypes_to_save):

                tensor_idx = class_idx * trained_model.num_prototypes_per_class + p_idx

                z_p_norm = zp_before[tensor_idx]
                z_p_denorm = (z_p_norm * std) + mean
                
                z_p_batch = z_p_denorm.to(device)
                waveform = diff_model.sample(z_p_batch)

                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.cpu().numpy()
                    
                wave_int16 = np.int16(waveform * 32767)
                
                filename = f"zeroshot_kmeans_prototype_{p_idx}.wav"
                filepath = os.path.join(class_dir, filename)
                
                wavfile.write(filepath, 24000, wave_int16)
                
    print("Zero-Shot K-Means prototypes generated.")

def find_and_save_closest_samples(trained_model, diff_model, dataloader, num_prototypes, output_dir="medley_experiments", device="cuda"):
    trained_model.eval()
    trained_model.to(device)
    diff_model.model.eval()
    diff_model.model.to(device)

    closest_dir = os.path.join(output_dir, "closest_real_samples")
    os.makedirs(closest_dir, exist_ok=True)
    
    print("\nExtracting real samples from dataloader to find closest matches...")
    all_zx = []
    all_labels = []
    with torch.no_grad():
        for z_x, labels in tqdm(dataloader, desc="Loading Audio Features"):
            all_zx.append(z_x.to(device))
            all_labels.append(labels.to(device))

    all_zx_tensor = torch.cat(all_zx, dim=0) 
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    with torch.no_grad():
        for class_idx in range(trained_model.num_classes):
            instr_name = MEDLEY_INSTRUMENTS.get(class_idx, f"class_{class_idx}")
            class_dir = os.path.join(closest_dir, f"{class_idx}_{instr_name}")
            os.makedirs(class_dir, exist_ok=True)

            class_mask = (all_labels_tensor == class_idx)
            class_zx_tensor = all_zx_tensor[class_mask]
            
            for p_idx in range(num_prototypes):
                z_p_norm = trained_model.get_prototype(class_idx, p_idx, projected=True, denormalize=False)
                distances = torch.cdist(class_zx_tensor, z_p_norm.unsqueeze(0))
                closest_idx = torch.argmin(distances).item()
                
                closest_zx_norm = all_zx_tensor[closest_idx]
                closest_zx_denorm = (closest_zx_norm * trained_model.data_std.to(device)) + trained_model.data_mean.to(device)
                
                waveform = diff_model.sample(closest_zx_denorm.to(device))
                wave_int16 = np.int16(waveform * 32767)
                
                filepath = os.path.join(class_dir, f"closest_real_sample_to_p{p_idx}.wav")
                wavfile.write(filepath, 24000, wave_int16)
                
    print("Closest real samples generated.")

def check_prototype_collapse(trained_model, dataloader, device="cuda"):
    trained_model.eval()
    trained_model.to(device)
    
    all_zx = []
    all_labels = []
    
    with torch.no_grad():
        for z_x, labels in tqdm(dataloader, desc="Loading Audio Features"):
            all_zx.append(z_x.to(device))
            all_labels.append(labels.to(device))
            
    all_zx_tensor = torch.cat(all_zx, dim=0) 
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    num_classes = trained_model.num_classes
    num_prototypes = trained_model.num_prototypes_per_class
    
    target_counts = defaultdict(list)
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            instr_name = MEDLEY_INSTRUMENTS.get(class_idx, f"class_{class_idx}") if 'MEDLEY_INSTRUMENTS' in globals() else f"class_{class_idx}"
            
            class_indices = torch.where(all_labels_tensor == class_idx)[0]
            class_zx_tensor = all_zx_tensor[class_indices]
            
            if len(class_zx_tensor) == 0:
                continue
                
            for p_idx in range(num_prototypes):
                z_p_norm = trained_model.get_prototype(class_idx, p_idx, projected=True, denormalize=False)
                
                distances = torch.cdist(class_zx_tensor, z_p_norm.unsqueeze(0))
                closest_local_idx = torch.argmin(distances).item()
                closest_global_idx = class_indices[closest_local_idx].item()
                
                target_counts[closest_global_idx].append(f"{instr_name}_P{p_idx}")
                
    total_prototypes = num_classes * num_prototypes
    unique_targets = len(target_counts)
    
    print("\n" + "="*50)
    print("PROTOTYPE COLLAPSE REPORT")
    print("="*50)
    print(f"Total Prototypes         : {total_prototypes}")
    print(f"Unique Nearest Neighbors : {unique_targets}")
    
    for idx, proto_list in target_counts.items():
        if len(proto_list) > 1:
            print(f"- Real Sample #{idx:<4} shared by {len(proto_list)} prototypes:")
            print(f"  -> {', '.join(proto_list)}")
            
    print("="*50 + "\n")
    return target_counts

def plot_tsne(latent_file_path, output_image_path="saves/tsne_visualization.png"):
    
    print(f"Loading latent data from {latent_file_path} for t-SNE...")
    if not os.path.exists(latent_file_path):
        print(f"Error: {latent_file_path} not found. Run training script first.")
        return
        
    data = torch.load(latent_file_path, map_location="cpu")
    zp_after_tensor = data['zp_after']
    
    mean_div = compute_mean_diversity(zp_after_tensor, num_classes=8)
    diversity_text = f"Mean Prototype Diversity (L2): {mean_div:.4f}" if mean_div is not None else "Prototype Diversity: N/A (M=1)"

    metrics_file = "sonification_metrics.json"
    
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = []
        
    metrics_data.append({
        "M": args.num_prototypes_per_class,
        "Lambda": args.lambda_weight,
        "Diversity_L2": mean_div if mean_div is not None else 0.0
    })
    
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=4)

    zx = data['zx'].numpy()            
    labels = data['labels'].numpy()    
    zp_before = data['zp_before'].numpy() 
    zp_after = zp_after_tensor.numpy()   
    
    all_data = np.vstack([zx, zp_before, zp_after])
    
    print("Running t-SNE algorithm")
    tsne = TSNE(n_components=2, perplexity=30, random_state=888)
    embedded_data = tsne.fit_transform(all_data)
    
    n_zx = len(zx)
    n_zp = len(zp_before)
    num_prototypes_per_class = n_zp // 8
    
    zx_emb = embedded_data[:n_zx]
    zp_before_emb = embedded_data[n_zx : n_zx + n_zp]
    zp_after_emb = embedded_data[n_zx + n_zp :]
    
    plt.figure(figsize=(12, 10))
    #colors = plt.cm.tab10(np.linspace(0, 1, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 8))
    
    for c in range(8):
        mask = (labels == c)
        plt.scatter(zx_emb[mask, 0], zx_emb[mask, 1], c=[colors[c]], alpha=0.6, s=10, label=MEDLEY_INSTRUMENTS.get(c, f"class_{c}"))

    for c in range(8):
        base_color = colors[c]
        dark_color = (base_color[0] * 0.8, base_color[1] * 0.8, base_color[2] * 0.8, 1.0)
        
        start_idx = c * num_prototypes_per_class
        end_idx = start_idx + num_prototypes_per_class
        
        label_before = 'Prototypes (Before - K-Means)' if c == 0 else "_nolegend_"
        label_after = 'Prototypes (After - Trained)' if c == 0 else "_nolegend_"

        plt.scatter(zp_before_emb[start_idx:end_idx, 0], zp_before_emb[start_idx:end_idx, 1], 
                    c=[dark_color], marker='X', s=100, edgecolors='white', zorder=5, label=label_before)
        plt.scatter(zp_after_emb[start_idx:end_idx, 0], zp_after_emb[start_idx:end_idx, 1], 
                    c=[dark_color], marker='*', s=150, edgecolors='black', zorder=6, label=label_after)

    for i in range(n_zp):
        plt.arrow(zp_before_emb[i, 0], zp_before_emb[i, 1], 
                  zp_after_emb[i, 0] - zp_before_emb[i, 0], 
                  zp_after_emb[i, 1] - zp_before_emb[i, 1], 
                  color='grey', alpha=0.6, 
                  head_width=0.4, head_length=0.4,
                  length_includes_head=True, zorder=4)
    
    plt.title(f"t-SNE Latent Space Visualization: Prototype Movement\n{diversity_text}", fontsize=14, pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    
    print(f"t-SNE visualization saved to {output_image_path} | {diversity_text}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_weight', type=float, required=True)
    parser.add_argument('--num_prototypes_per_class', type=int, required=True)
    parser.add_argument('--num_prototypes_to_save', type=int, default=1)
    parser.add_argument('--compute_closest', action='store_true', help="If set, also finds and synthesizes the closest real samples")
    parser.add_argument('--compute_zero_shot', action='store_true', help="If set, also synthesizes the zero shot K-Means centroids")
    args = parser.parse_args()

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cpu")
    print(f"Using device: {device}")
    if not torch.cuda.is_available():
        os.environ["ACCELERATOR"] = "cpu"

    MODEL_PATH = f"saves/model_M{args.num_prototypes_per_class}_L{args.lambda_weight}.pth"
    LATENT_PATH = f"saves/model_M{args.num_prototypes_per_class}_L{args.lambda_weight}_latent_tensors.pt"
    OUTPUT_DIR = f"medley_experiments/model_M{args.num_prototypes_per_class}_L{args.lambda_weight}"
    TSNE_OUTPUT = f"saves/tsne_M{args.num_prototypes_per_class}_L{args.lambda_weight}.png"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}. Skipping.")
        exit(1)
        
    model = PrototypicalNetwork(num_classes=8, num_prototypes_per_class=args.num_prototypes_per_class).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    diff_model = hub.load_model("DiffTransformerAE2L8L1CLS-4s")
    if hasattr(diff_model, 'model'):
        diff_model.model.to(device)
    elif hasattr(diff_model, 'to'):
        diff_model.to(device)

    generate_and_save_prototypes(
        trained_model=model,
        diff_model=diff_model,
        num_prototypes=args.num_prototypes_to_save,
        output_dir=OUTPUT_DIR,
        device=device
    )

    if args.compute_zero_shot:
        generate_zeroshot_kmeans_audio(
        trained_model=model,
        diff_model=diff_model,
        latent_file_path=LATENT_PATH,
        num_prototypes_to_save=args.num_prototypes_to_save,
        output_dir=OUTPUT_DIR,
        device=device
        )

    if args.compute_closest:
        train_loader, _, _, _, _ = create_dataloaders(
            features_file='data/medley_features.pt',
            batch_size=256,
            num_workers=0
        )

        find_and_save_closest_samples(
            trained_model=model,
            diff_model=diff_model,
            dataloader=train_loader,
            num_prototypes=args.num_prototypes_to_save, 
            output_dir=OUTPUT_DIR,
            device=device
        )
       
    plot_tsne(latent_file_path=LATENT_PATH, output_image_path=TSNE_OUTPUT)

    train_loader, _, _, _, _ = create_dataloaders(
            features_file='data/medley_features.pt',
            batch_size=256,
            num_workers=0
        )

    check_prototype_collapse(model, train_loader, device=device)