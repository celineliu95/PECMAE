import os
import torch
from scipy.io import wavfile
from tqdm import tqdm
from model import PrototypicalNetwork
import numpy as np
from encodecmae_to_wav.hub import load_model

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
                
    print("Done.")

if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = PrototypicalNetwork(num_classes=8, num_prototypes_per_class=5)
    model.load_state_dict(torch.load("saves/best_prototypical_network.pth", map_location=device))
    diff_model = load_model("DiffTransformerAE2L8L1CLS-4s")

    generate_and_save_prototypes(
        trained_model=model,
        diff_model=diff_model,
        num_prototypes=1,
        output_dir="medley_prototypes",
        device=device
    )