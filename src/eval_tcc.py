import os
import glob
import re
import json
import librosa

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from model import PrototypicalNetwork
import encodecmae_to_wav.hub as hub

_original_get_model = hub.get_model
def patched_get_model(*args, **kwargs):
    kwargs.pop('processor', None)
    return _original_get_model(*args, **kwargs)
hub.get_model = patched_get_model

def evaluate_all_models_tcc(experiments_dir="medley_experiments", saves_dir="saves", device="cpu"):
    
    diff_model = hub.load_model("DiffTransformerAE2L8L1CLS-4s")
    if hasattr(diff_model, 'model'):
        diff_model.model.to(device)
        diff_model.model.eval()
    elif hasattr(diff_model, 'to'):
        diff_model.to(device)
        diff_model.eval()
        
    results = []
    
    model_dirs = glob.glob(os.path.join(experiments_dir, "model_M*_L*"))
    
    for model_dir in tqdm(model_dirs, desc="Evaluating Models"):
        dirname = os.path.basename(model_dir)
        match = re.search(r"model_M(\d+)_L([0-9.]+)", dirname)
        if not match:
            continue
            
        m_val = int(match.group(1))
        lam_val = float(match.group(2))
        
        pth_path = os.path.join(saves_dir, f"{dirname}.pth")
        if not os.path.exists(pth_path):
            print(f"\nMissing weights for {dirname}. Skipping.")
            continue
            
        classif_model = PrototypicalNetwork(num_classes=8, num_prototypes_per_class=m_val).to(device)
        classif_model.load_state_dict(torch.load(pth_path, map_location=device))
        classif_model.eval()
        
        tcc_scores = []
        class_dirs = glob.glob(os.path.join(model_dir, "*_*")) 
        for c_dir in class_dirs:
            folder_name = os.path.basename(c_dir)
            true_class_idx = int(folder_name.split("_")[0])
            
            wav_files = glob.glob(os.path.join(c_dir, "*.wav"))
            
            proto_files = [f for f in wav_files if "prototype_" in os.path.basename(f)]
            for wav_path in proto_files:
                x, fs = librosa.load(wav_path, sr=24000)
                
                with torch.no_grad():
                    codes = diff_model.encode(x).to(device) 
                    codes_norm = (codes - classif_model.data_mean) / classif_model.data_std
                    
                    logits, _, _ = classif_model(codes_norm)
                    probs = F.softmax(logits, dim=1)
                    
                    confidence = probs[0, true_class_idx].item()
                    tcc_scores.append(confidence)
                    
        if tcc_scores:
            mean_tcc = np.mean(tcc_scores)
            results.append({
                "M": m_val,
                "Lambda": lam_val,
                "Mean_TCC": mean_tcc
            })

    output_json = "tcc_metrics.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    
    for res in sorted(results, key=lambda x: (x['M'], x['Lambda'])):
        print(f"M={res['M']:2d} | Lam={res['Lambda']:.2f} -> Mean TCC = {res['Mean_TCC']*100:.2f}%")

if __name__ == "__main__":
    evaluate_all_models_tcc()