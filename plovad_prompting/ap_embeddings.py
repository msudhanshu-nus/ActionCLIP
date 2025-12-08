import numpy as np
import pandas as pd
import torch

def load_plovad_text_features(label_csv, npy_path, device):
    labels = pd.read_csv(label_csv)['name'].tolist()  # ['bleeding', 'mechanical_injury', 'thermal_injury']
    feats = torch.from_numpy(np.load(npy_path)).float()  # (num_classes, 512)
    if feats.shape[0] != len(labels):
        raise ValueError(f"Class count mismatch: {feats.shape[0]} vs {len(labels)}")
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.to(device)