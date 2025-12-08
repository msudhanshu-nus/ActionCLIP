#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert llm_descriptions (dict[label] -> list[prompt]) from a .py file
into CLIP text embeddings, in a PLOVAD-style fashion.

Output:
  - .npy: shape [num_classes, D], row i corresponds to labels[i]
  - .txt: class names in the same order as in the .npy
"""

import os
import argparse
import importlib.util

import numpy as np
import torch
import clip


def load_llm_descriptions(py_file: str, var_name: str = "llm_descriptions"):
    """
    Dynamically import a .py file and return the variable `var_name`.
    Expected type: dict[str, list[str]]
    """
    spec = importlib.util.spec_from_file_location("prompt_module", py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, var_name):
        raise AttributeError(
            f"{var_name} not found in {py_file}. "
            f"Make sure you defined `{var_name} = {{...}}`."
        )

    desc = getattr(module, var_name)
    if not isinstance(desc, dict):
        raise TypeError(f"{var_name} must be a dict, got {type(desc)}")

    # Optional: enforce deterministic label order (sorted)
    labels = sorted(desc.keys())
    return desc, labels


def prompt2vec_from_py(
    py_file: str,
    out_npy: str,
    out_labels_txt: str,
    clip_backbone: str = "ViT-B/16",
    var_name: str = "llm_descriptions",
):
    """
    PLOVAD-style:
      - load CLIP (frozen),
      - for each label: encode all its prompts,
      - average embeddings per label,
      - concat to [num_labels, D] and save.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP and freeze (same spirit as PLOVAD)
    clip_model, _ = clip.load(clip_backbone, device=device, jit=False)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    # Load descriptions from .py
    llm_desc, labels = load_llm_descriptions(py_file, var_name=var_name)
    print("Labels (order used for rows):", labels)

    clip_feat = torch.zeros(0, clip_model.text_projection.shape[1], device=device)

    for label in labels:
        prompts = llm_desc[label]
        if not isinstance(prompts, (list, tuple)):
            raise TypeError(
                f"Value for label '{label}' must be a list of strings, got {type(prompts)}"
            )

        # Build prompt list
        prompt_list = [str(p) for p in prompts if isinstance(p, str)]
        if len(prompt_list) == 0:
            raise ValueError(f"No valid string prompts found for label '{label}'")

        # Tokenize & encode
        tokens = clip.tokenize(prompt_list).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)  # [N_prompts, D]
            # Optional: normalize each prompt embedding first
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_mean = emb.mean(dim=0)           # [D]

        clip_feat = torch.cat((clip_feat, emb_mean.view(1, -1)), dim=0)
        print(f"{label}: {len(prompt_list)} prompts -> embedding {emb_mean.shape}")

    print(f"Final embedding shape: {clip_feat.shape}")  # [num_classes, D]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)

    # Save .npy (PLOVAD style: just a 2D array)
    np.save(out_npy, clip_feat.detach().cpu().numpy())
    print(f"Saved embeddings to: {out_npy}")

    # Save label order alongside
    with open(out_labels_txt, "w") as f:
        for lbl in labels:
            f.write(lbl + "\n")
    print(f"Saved label order to: {out_labels_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PromptExtract from .py llm_descriptions")
    parser.add_argument(
        "--py_file",
        type=str,
        default="llm_descriptions.py",
        help="Path to .py file containing `llm_descriptions` dict",
    )
    parser.add_argument(
        "--var_name",
        type=str,
        default="llm_descriptions",
        help="Variable name inside the .py that holds the dict",
    )
    parser.add_argument(
        "--out_npy",
        type=str,
        default="./prompt_feature/multibypass-llm-prompts.npy",
        help="Output .npy file for prompt embeddings",
    )
    parser.add_argument(
        "--out_labels",
        type=str,
        default="./prompt_feature/multibypass-llm-prompts_labels.txt",
        help="Output .txt with class labels in row order",
    )
    parser.add_argument(
        "--clip_backbone",
        type=str,
        default="ViT-B/16",
        help="CLIP backbone (e.g., ViT-B/16, RN50, etc.)",
    )
    args = parser.parse_args()

    torch.cuda.set_device(0) if torch.cuda.is_available() else None

    prompt2vec_from_py(
        py_file=args.py_file,
        out_npy=args.out_npy,
        out_labels_txt=args.out_labels,
        clip_backbone=args.clip_backbone,
        var_name=args.var_name,
    )
