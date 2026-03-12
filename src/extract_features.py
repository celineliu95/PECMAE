import sys
import argparse
import torch
import librosa
import os
import pandas as pd
from tqdm import tqdm

if '/content/encodecmae' not in sys.path:
    sys.path.insert(0, '/content/encodecmae')

from encodecmae_to_wav.hub import load_model

def extract_and_save_features(audio_filepaths, labels, diff_model, save_path='features.pt'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diff_model.model.eval()
    diff_model.model.to(device)
    
    all_codes = []
    all_labels = []
    
    with torch.no_grad():
        for filepath, label in tqdm(zip(audio_filepaths, labels), total=len(audio_filepaths)):
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
                
            x, fs = librosa.core.load(filepath, sr=24000)
            codes = diff_model.encode(x)
            
            all_codes.append(codes.cpu())
            all_labels.append(label)
            
    if not all_codes:
        print("No features extracted. Please check the file paths.")
        return

    X_tensor = torch.cat(all_codes, dim=0) 
    y_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    torch.save({'X': X_tensor, 'y': y_tensor}, save_path)
    print(f"Saved in {save_path} (Shape: {X_tensor.shape})")

def parse_medley_csv(csv_path, audio_dir):
    df = pd.read_csv(csv_path)
    audio_filepaths = []
    labels = []
    
    for _, row in df.iterrows():
        subset = row['subset']
        instrument_id = row['instrument_id']
        uuid4 = row['uuid4']
        
        filename = f"Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav"
        filepath = os.path.join(audio_dir, filename)
        
        audio_filepaths.append(filepath)
        labels.append(instrument_id)
        
    return audio_filepaths, labels

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--audio_files', nargs='+', help='List of audio file paths')
    parser.add_argument('--labels', nargs='+', type=int, help='Corresponding list of integer labels')
    
    parser.add_argument('--csv_path', type=str, help='Path to the Medley-Solos-DB metadata CSV')
    parser.add_argument('--audio_dir', type=str, help='Directory containing the audio files')
    
    parser.add_argument('--save_path', default='data/features.pt', help='Path to save the extracted features')
    parser.add_argument('--model_name', default='DiffTransformerAE2L8L1CLS-4s', help='Name of the pre-trained model to use for feature extraction')

    args = parser.parse_args()

    paths_wav = []
    labels_int = []

    if args.csv_path and args.audio_dir:
        paths_wav, labels_int = parse_medley_csv(args.csv_path, args.audio_dir)
    elif args.audio_files and args.labels:
        if len(args.audio_files) != len(args.labels):
            raise ValueError("The number of audio files must correspond to the number of labels.")
        paths_wav = args.audio_files
        labels_int = args.labels
    else:
        raise ValueError("Specify either (--audio_files AND --labels) or (--csv_path AND --audio_dir).")

    print(f"Loading model {args.model_name}...")
    diff_model = load_model(args.model_name)
    
    extract_and_save_features(paths_wav, labels_int, diff_model, args.save_path)