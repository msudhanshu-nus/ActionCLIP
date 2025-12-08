# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.optim as optim
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def _optimizer(config, model, fusion_model, extra_params=None):
    """Build optimizer with optional extra parameter groups (e.g., prompts)."""
    extra_params = list(extra_params) if extra_params else []

    def _dedup(params, taken):
        # Keep parameter order but drop references that were already added.
        filtered = []
        for p in params:
            if id(p) in taken:
                continue
            taken.add(id(p))
            filtered.append(p)
        return filtered

    taken = set()

    if config.solver.optim == 'adam':
        model_params = _dedup(model.parameters(), taken)
        fusion_params = _dedup(fusion_model.parameters(), taken)
        prompt_params = _dedup(extra_params, taken)
        param_groups = [
            {'params': model_params},
            {'params': fusion_params, 'lr': config.solver.lr * config.solver.f_ratio},
        ]
        if prompt_params:
            param_groups.append({'params': prompt_params})
        optimizer = optim.Adam(
            param_groups,
            lr=config.solver.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.2
        )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        print('Adam')
    elif config.solver.optim == 'sgd':
        model_params = _dedup(model.parameters(), taken)
        fusion_params = _dedup(fusion_model.parameters(), taken)
        prompt_params = _dedup(extra_params, taken)
        param_groups = [
            {'params': model_params},
            {'params': fusion_params, 'lr': config.solver.lr * config.solver.f_ratio},
        ]
        if prompt_params:
            param_groups.append({'params': prompt_params})
        optimizer = optim.SGD(
            param_groups,
            config.solver.lr,
            momentum=config.solver.momentum,
            weight_decay=config.solver.weight_decay
        )
        print('SGD')
    elif config.solver.optim == 'adamw':
        visual_params = list(model.visual.parameters())
        visual_param_ids = {id(p) for p in visual_params}
        text_params = [p for p in model.parameters() if id(p) not in visual_param_ids]

        visual_params = _dedup(visual_params, taken)
        text_params = _dedup(text_params, taken)
        fusion_params = _dedup(fusion_model.parameters(), taken)
        prompt_params = _dedup(extra_params, taken)

        param_groups = [
            {'params': text_params},
            {'params': visual_params, 'lr': config.solver.lr * config.solver.ratio},
            {'params': fusion_params, 'lr': config.solver.lr * config.solver.f_ratio},
        ]
        if prompt_params:
            param_groups.append({'params': prompt_params})

        optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.98),
            lr=config.solver.lr,
            eps=1e-8,
            weight_decay=config.solver.weight_decay
        )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.solver.optim))
    return optimizer

def _lr_scheduler(config,optimizer):
    if config.solver.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            config.solver.epochs,
            warmup_epochs=config.solver.lr_warmup_step
        )
    elif config.solver.type == 'multistep':
        if isinstance(config.solver.lr_decay_step, list):
            milestones = config.solver.lr_decay_step
        elif isinstance(config.solver.lr_decay_step, int):
            milestones = [
                config.solver.lr_decay_step * (i + 1)
                for i in range(config.solver.epochs //
                               config.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(config.solver.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=config.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(config.solver.type))
    return lr_scheduler
