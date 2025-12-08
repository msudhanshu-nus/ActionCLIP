# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import csv
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from plovad_prompting.ap_embeddings import load_plovad_text_features 
from utils.learnable_prompts_plovad import *

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, text_features=None, threshold=0.5, prompt_source=None, ap_features=None, dp_features=None, label_names=None, collect_predictions=False):
    """Multi-label validation with sigmoid probabilities and macro F1.

    When collect_predictions is True, per-clip predictions are returned for CSV export.
    """
    model.eval()
    fusion_model.eval()
    prompt_source = prompt_source or ""

    # Allow scalar or per-class thresholds
    def get_threshold_tensor(thresh_value, num_classes, device):
        if isinstance(thresh_value, (list, tuple, np.ndarray)):
            t = torch.tensor(thresh_value, device=device, dtype=torch.float32)
            if t.numel() != num_classes:
                raise ValueError(f"Threshold length {t.numel()} does not match num_classes {num_classes}")
            return t
        return torch.tensor(thresh_value, device=device, dtype=torch.float32).expand(num_classes)

    severity_weights =  [0.02, 0.06, 0.12, 0.19, 0.26, 0.33]
    num_severity_levels = 6

    with torch.no_grad():
        if prompt_source == "plovad_both":
            if ap_features is None or dp_features is None:
                raise ValueError("plovad_both validation requires both ap_features and dp_features")
            ap_features = ap_features.to(device)
            dp_features = dp_features.to(device)
            ap_features = ap_features / ap_features.norm(dim=-1, keepdim=True)
            dp_features = dp_features / dp_features.norm(dim=-1, keepdim=True)
            num_classes = ap_features.size(0)
        else:
            if text_features is None:
                text_inputs = classes.to(device)
                raw_text_features = model.encode_text(text_inputs)
                text_features = raw_text_features.view(num_text_aug, -1, raw_text_features.size(-1)).mean(dim=0)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            num_classes = text_features.size(0)
            text_features = text_features.to(device=device)
        thresh_tensor = get_threshold_tensor(threshold, num_classes, device)

        all_targets = []
        all_preds = []
        all_probs = []
        all_severities = []
        all_clip_ids = [] if collect_predictions else None

        for batch in tqdm(val_loader):
            severities = None
            clip_paths = None
            if len(batch) == 4:
                image, targets, severities, clip_paths = batch
            elif len(batch) == 3:
                image, targets, third = batch
                if torch.is_tensor(third):
                    severities = third
                else:
                    clip_paths = third
            else:
                image, targets = batch
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            targets = targets.to(device).float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            targets = targets[:, :num_classes]
            if severities is not None:
                severities = severities.to(device)
                if severities.dim() == 1:
                    severities = severities.unsqueeze(1)
                severities = severities[:, :num_classes]

            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)

            v = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()

            if prompt_source == "plovad_both":
                tf_ap = ap_features.to(device=device, dtype=v.dtype)
                tf_dp = dp_features.to(device=device, dtype=v.dtype)
                logits = 0.5 * (
                    logit_scale * (v @ tf_ap.t()) +
                    logit_scale * (v @ tf_dp.t())
                )
            else:
                tf = text_features.to(device=device, dtype=v.dtype)  # match image feature dtype (fp16/fp32)
                logits = logit_scale * (v @ tf.t())
            probs = logits.sigmoid()
            preds = (probs >= thresh_tensor).float()

            all_targets.append(targets.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            if severities is not None:
                all_severities.append(severities.cpu())
            if collect_predictions:
                if clip_paths is None:
                    clip_batch = [f"idx_{len(all_clip_ids) + i}" for i in range(targets.size(0))]
                else:
                    if not isinstance(clip_paths, (list, tuple)):
                        clip_paths = list(clip_paths)
                    clip_batch = [os.path.basename(str(p)) for p in clip_paths]
                    if len(clip_batch) != targets.size(0):
                        raise ValueError(f"Mismatch between clip paths ({len(clip_batch)}) and targets ({targets.size(0)})")
                all_clip_ids.extend(clip_batch)



    targets_np = torch.cat(all_targets).numpy()
    preds_np = torch.cat(all_preds).numpy()
    probs_np = torch.cat(all_probs).numpy()
    severities_np = torch.cat(all_severities).numpy() if len(all_severities) > 0 else None

    per_class_f1 = []
    for i in range(num_classes):
        per_class_f1.append(f1_score(targets_np[:, i], preds_np[:, i], zero_division=0))
    macro_f1 = float(np.mean(per_class_f1))

    per_class_ap = []
    for i in range(num_classes):
        y_true = targets_np[:, i]
        y_score = probs_np[:, i]
        uniq = np.unique(y_true)
        if uniq.size == 1:
            per_class_ap.append(float(uniq[0]))
            continue
        ap_i = average_precision_score(y_true, y_score)
        if np.isnan(ap_i):
            ap_i = 0.0
        per_class_ap.append(float(ap_i))
    macro_ap = float(np.mean(per_class_ap))

    # Severity-weighted F1 per class and overall (mean across classes).
    severity_weighted = None
    per_class_severity_f1 = None
    if severities_np is not None:
        severities_np = np.nan_to_num(severities_np, nan=0.0, posinf=0.0, neginf=0.0)
        severities_np[severities_np < 0] = 0  # Guard against malformed labels
        per_class_severity_f1 = []
        for i in range(num_classes):
            sev_levels = severities_np[:, i]
            t_i = targets_np[:, i].astype(np.int32)
            p_i = preds_np[:, i].astype(np.int32)
            level_f1 = []
            for lvl in range(num_severity_levels):
                mask = sev_levels == lvl
                n = int(mask.sum())
                if n == 0:
                    level_f1.append(0.0)
                    continue

                y_true = t_i[mask]
                y_pred = p_i[mask]
                level_f1.append(f1_score(y_true, y_pred, zero_division=0))

            level_f1 = np.array(level_f1, dtype=np.float32)
            per_class_severity_f1.append(float(np.dot(severity_weights, level_f1)))
        severity_weighted = float(np.mean(per_class_severity_f1))

    log_data = {"val_macro_f1": macro_f1}
    for i, f1_val in enumerate(per_class_f1):
        log_data[f"val_f1_class_{i}"] = f1_val
    log_data["val_macro_ap"] = macro_ap
    for i, ap_val in enumerate(per_class_ap):
        log_data[f"val_ap_class_{i}"] = ap_val
    if severity_weighted is not None:
        log_data["val_severity_weighted_f1"] = severity_weighted
        for i, sev_f1 in enumerate(per_class_severity_f1):
            log_data[f"val_severity_weighted_f1_class_{i}"] = sev_f1
    wandb.log(log_data)

    metrics = {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "macro_ap": macro_ap,
        "per_class_ap": per_class_ap,
        "severity_weighted_f1": severity_weighted,
        "per_class_severity_weighted_f1": per_class_severity_f1,
    }

    predictions = None
    if collect_predictions:
        predictions = {
            "clip_ids": all_clip_ids,
            "targets": targets_np,
            "probs": probs_np,
            "preds": preds_np,
            "severities": severities_np,
        }

    print('Epoch: [{}/{}]: Macro F1: {:.2f} | Macro AP: {:.2f}'.format(epoch, config.solver.epochs, macro_f1, macro_ap))
    return (metrics, predictions) if collect_predictions else metrics


def save_predictions_csv(predictions, label_names, output_path):
    """Write clip-level predictions to CSV with severity columns."""
    clip_ids = predictions.get("clip_ids")
    targets = predictions["targets"]
    probs = predictions["probs"]
    preds = predictions["preds"]
    severities = predictions.get("severities")

    if severities is None:
        raise ValueError("Predictions missing severity values; cannot write severity columns.")

    shapes = (targets.shape, probs.shape, preds.shape, severities.shape)
    if not (targets.shape == probs.shape == preds.shape == severities.shape):
        raise ValueError(f"Targets, probs, preds, and severities must share shape; got {shapes}.")

    num_samples, num_classes = targets.shape
    if clip_ids is not None and len(clip_ids) != num_samples:
        raise ValueError(f"clip_ids length {len(clip_ids)} does not match number of samples {num_samples}.")

    expected_names = ["bleeding", "mechanical_injury", "thermal_injury"]
    safe_names = (
        [str(name).replace(" ", "_") for name in label_names]
        if label_names and len(label_names) == num_classes
        else [f"class_{i}" for i in range(num_classes)]
    )
    name_to_idx = {name.lower(): idx for idx, name in enumerate(safe_names)}
    missing = [name for name in expected_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Missing expected class name(s) in label_names: {', '.join(missing)}")
    class_indices = [name_to_idx[name] for name in expected_names]

    header = [
        "clip_id",
        "bleeding_gt",
        "bleeding_severity",
        "mechanical_injury_gt",
        "mechanical_injury_severity",
        "thermal_injury_gt",
        "thermal_injury_severity",
        "prob_bleeding",
        "pred_bleeding",
        "prob_mechanical_injury",
        "pred_mechanical_injury",
        "prob_thermal_injury",
        "pred_thermal_injury",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for idx in range(num_samples):
            clip_val = clip_ids[idx] if clip_ids is not None else f"idx_{idx}"
            row = [clip_val]

            for j in class_indices:
                row.append(int(targets[idx, j]))
                row.append(int(severities[idx, j]))
            for j in class_indices:
                row.append(float(probs[idx, j]))
                row.append(int(preds[idx, j]))
            writer.writerow(row)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    parser.add_argument(
        '--output_root',
        default='./exp',
        help='Root directory where checkpoints/config copies are saved',
    )
    parser.add_argument(
        '--log_root',
        default=None,
        help='Directory to store wandb logs (defaults to wandb default when unset)',
    )
    parser.add_argument(
        '--predictions_dir',
        default='./predictions',
        help='Directory to save prediction CSVs',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Decision threshold for sigmoid probabilities when computing F1 (overridden by config.data.thresholds when provided)',
    )
    parser.add_argument('--fold', type=int, default=None, help='Fold index for cross-validation (e.g., 0-4)')
    parser.add_argument('--split', choices=['val', 'test'], default='val', help='Which split list to use')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = DotMap(config)

    if args.fold is not None:
        dataset_name = getattr(config.data, "dataset", "mby140")
        if args.split == "val":
            config.data.val_list = f'lists/{dataset_name}/fold{args.fold}/mb_val.txt'
        else:
            config.data.val_list = f'lists/{dataset_name}/fold{args.fold}/mb_test.txt'
        config.data.dataset = dataset_name
        pretrain_template = getattr(config, "pretrain", None)
        if isinstance(pretrain_template, str):
            try:
                config.pretrain = pretrain_template.format(fold=args.fold, dataset=dataset_name)
            except KeyError:
                config.pretrain = pretrain_template
        if (not getattr(config, "pretrain", None)) and getattr(config, "pretrain_root", None):
            config.pretrain = os.path.join(
                config.pretrain_root,
                f"{dataset_name}_fold{args.fold}",
                f"fold{args.fold}",
                "model_best.pt",
            )

    working_dir = os.path.join(
        args.output_root,
        config['network']['type'],
        config['network']['arch'],
        f"{config['data']['dataset']}_fold{args.fold}" if args.fold is not None else config['data']['dataset'],
        args.log_time,
    )

    if args.log_root is not None:
        Path(args.log_root).mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=config['network']['type'],
        name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'], config['data']['dataset']),
        dir=args.log_root,
    )
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    prompt_source = getattr(config.data, "prompt_source", "")
    use_learnable = prompt_source in ("plovad_learnable", "plovad_both")
    prompt_encoder = PlovadLearnablePrompt(
          model,
          prefix=getattr(config.data, "prompt_prefix", 2),
          postfix=getattr(config.data, "prompt_postfix", 2),
          std_init=getattr(config.data, "prompt_std_init", 0.02),
      ).to(device) if use_learnable else None
    
    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).to(device)
    model_image = torch.nn.DataParallel(model_image).to(device)
    fusion_model = torch.nn.DataParallel(fusion_model).to(device)
    wandb.watch(model)
    wandb.watch(fusion_model)

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=False, include_severity=True, include_path=True)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)
    label_names = [
        str(c[1] if isinstance(c, (list, tuple)) and len(c) > 1 else c)
        for c in val_data.classes
    ]

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug = None, 1  # placeholders when using precomputed/learnable prompts
    dp_features = None
    ap_features = None

    if prompt_source == "plovad":
        text_features = load_plovad_text_features(
            config.data.label_list,
            "plovad_prompting/ap_prompts_mb.npy",
            device,
        )
        num_classes = text_features.size(0)
        classes, num_text_aug = None, 1  # placeholders; text_features is final
    elif prompt_source == "plovad_learnable":
        class_names = [c for _, c in val_data.classes]
        with torch.no_grad():
            text_features = prompt_encoder(class_names)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        num_classes = text_features.size(0)
    elif prompt_source == 'plovad_both': 
        ap_features = load_plovad_text_features(config.data.label_list, "plovad_prompting/ap_prompts_mb.npy", device)
        ap_features = ap_features / ap_features.norm(dim=-1, keepdim=True)
        class_names = [c for _, c in val_data.classes]  # use val_data in test.py
        with torch.no_grad():
            dp_features = prompt_encoder(class_names)
            dp_features = dp_features / dp_features.norm(dim=-1, keepdim=True)
        if ap_features.shape != dp_features.shape:
            raise ValueError(f"AP/DP shape mismatch: {ap_features.shape} vs {dp_features.shape}")
        num_classes = ap_features.size(0)
        text_features = None
    else:
        classes, num_text_aug, text_dict = text_prompt(val_data)
        with torch.no_grad():
            text_inputs = classes.to(device)
            raw = model_text(text_inputs)
            text_features = raw.view(num_text_aug, -1, raw.size(-1)).mean(0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        num_classes = text_features.size(0)

    thresholds_cfg = getattr(config.data, "thresholds", None)
    thresholds_value = thresholds_cfg if thresholds_cfg is not None else args.threshold
    if isinstance(thresholds_value, (list, tuple, np.ndarray)):
        if len(thresholds_value) != num_classes:
            raise ValueError(f"Config/data thresholds length {len(thresholds_value)} does not match num_classes {num_classes}")
        thresholds_value = [float(t) for t in thresholds_value]
    else:
        thresholds_value = float(thresholds_value)
    print(f"Using decision threshold(s): {thresholds_value}")

    # classes, num_text_aug, text_dict = text_prompt(val_data)
    # with torch.no_grad():
    #     text_inputs = classes.to(device)
    #     raw_text_features = model_text(text_inputs)
    #     text_features = raw_text_features.view(num_text_aug, -1, raw_text_features.size(-1)).mean(dim=0)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    best_prec1 = 0.0
    metrics, predictions = validate(
        start_epoch,
        val_loader,
        classes,
        device,
        model,
        fusion_model,
        config,
        num_text_aug,
        text_features=text_features,
        threshold=thresholds_value,
        prompt_source=prompt_source,
        ap_features=ap_features if prompt_source == "plovad_both" else None,
        dp_features=dp_features if prompt_source == "plovad_both" else None,
        collect_predictions=True,
    )

    predictions_dir = Path(args.predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    val_list_path = Path(config.data.val_list)
    file_stem = val_list_path.stem
    parent_stem = val_list_path.parent.name
    if parent_stem:
        file_stem = f"{parent_stem}_{file_stem}"
    output_csv = predictions_dir / f"{file_stem}_prediction.csv"
    if predictions is not None:
        save_predictions_csv(predictions, label_names, output_csv)
        print(f"Saved predictions to {output_csv}")

if __name__ == '__main__':
    main()
