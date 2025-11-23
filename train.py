# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import torch
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
# from utils.KLLoss import KLLoss
from torch.nn import BCEWithLogitsLoss
from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

def main():
    global args, best_prec1
    best_ap = 0.0
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
    parser.add_argument('--fold', type=int, default=None, help='Fold index for cross-validation (e.g., 0-4)')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = DotMap(config)

    # Optional fold override for cross-validation.
    if args.fold is not None:
        config.data.dataset = getattr(config.data, "dataset", "mby140")
        config.data.train_list = f'lists/{config.data.dataset}/fold{args.fold}/mb_train.txt'
        config.data.val_list = f'lists/{config.data.dataset}/fold{args.fold}/mb_val.txt'

    working_dir = os.path.join(
        args.output_root,
        config.network.type,
        config.network.arch,
        f"{config.data.dataset}_fold{args.fold}" if args.fold is not None else config.data.dataset,
        args.log_time,
    )

    if args.log_root is not None:
        Path(args.log_root).mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=config.network.type,
        name='{}_{}_{}_{}'.format(args.log_time, config.network.type, config.network.arch, config.data.dataset),
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
    shutil.copy('train.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch,device=device,jit=False, tsm=config.network.tsm, T=config.data.num_segments,dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,pretrain=config.network.init, joint = config.network.joint) #Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = visual_prompt(config.network.sim_header,clip_state_dict,config.data.num_segments)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(fusion_model)

    train_data = Action_DATASETS(config.data.train_list,config.data.label_list,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,random_shift=config.data.random_shift,
                       transform=transform_train)
    train_loader = DataLoader(train_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=True,pin_memory=False,drop_last=True)
    val_data = Action_DATASETS(config.data.val_list,config.data.label_list, random_shift=False,num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,
                       transform=transform_val, include_severity=True)
    val_loader = DataLoader(val_data,batch_size=config.data.batch_size,num_workers=config.data.workers,shuffle=False,pin_memory=False,drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(train_data)

    # Pre-compute averaged text embeddings per class across prompt augmentations.
    with torch.no_grad():
        text_inputs = classes.to(device)  # [num_text_aug * num_classes, token_len]
        raw_text_features = model_text(text_inputs)  # [num_text_aug * num_classes, D]
        text_features = raw_text_features.view(num_text_aug, -1, raw_text_features.size(-1)).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [num_classes, D]

    num_classes = text_features.shape[0]
    loss_cfg = getattr(config, "loss", None)
    if loss_cfg is not None and getattr(loss_cfg, "pos_weight", None) is not None:
        pos_weight = torch.tensor(loss_cfg.pos_weight, device=device, dtype=text_features.dtype)
    else:
        pos_weight = torch.ones(num_classes, device=device, dtype=text_features.dtype)
    bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = _optimizer(config, model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    if config.solver.evaluate:
        validate(start_epoch,val_loader, classes, device, model,fusion_model, config,num_text_aug, text_features)
        return

    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        for kkk,(images,targets) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1,config.data.num_segments,3)+images.size()[-2:])
            b,t,c,h,w = images.size()
            images= images.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            targets = targets.to(device).float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            targets = targets[:, :num_classes]

            image_embedding = model_image(images)
            image_embedding = image_embedding.view(b,t,-1)
            image_embedding = fusion_model(image_embedding)

            logit_scale = model.logit_scale.exp()
            v = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            logits = logit_scale * (v @ text_features.t())  # [B, num_classes]

            loss = bce_loss(logits, targets)
            wandb.log({"train_loss": loss})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        metrics = None
        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            metrics = validate(epoch,val_loader, classes, device, model,fusion_model, config,num_text_aug, text_features)
            prec1 = metrics["macro_f1"]
            ap = metrics["macro_ap"]

            is_best_f1 = prec1 > best_prec1
            is_best_ap = ap > best_ap
            best_prec1 = max(prec1, best_prec1)
            best_ap = max(ap, best_ap)
            print('Testing F1/AP: {}/{}  | AP: {}/{}'.format(prec1, best_prec1, ap, best_ap))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(epoch, model, fusion_model, optimizer, filename)
        if metrics is not None:
            if is_best_f1:
                best_saving(working_dir, epoch, model, fusion_model, optimizer)
            if is_best_ap:
                best_ap_saving(working_dir, epoch, model, fusion_model, optimizer)

if __name__ == '__main__':
    main()
