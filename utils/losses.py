import torch.nn.functional as F
import torch.nn as nn
import torch  # torch base module

class MultiLabelBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits: [B, 3], targets: [B, 3], multi-hot
        return self.bce(logits, targets)

# class MultiLabelASLoss(nn.Module):
#     def __init__(self, pos_weight=None):
#         super().__init__()
#         if pos_weight is not None:
#             self.bce = nn.ASL(pos_weight=pos_weight)
#         else:
#             self.bce = nn.ASL()

#     def forward(self, logits, targets):
#         # logits: [B, 3], targets: [B, 3], multi-hot
#         return self.bce(logits, targets)


class FocalLossMultiLabel(nn.Module):  # focal loss for multi-label problems
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", eps=1e-8):  # store parameters
        super().__init__()  # initialize parent nn.Module
        self.alpha = alpha  # per-class balance weights for positives
        self.gamma = gamma  # focusing parameter to down-weight easy examples
        self.reduction = reduction  # reduction mode: mean, sum, or none
        self.eps = eps  # numerical stability for logs and probs

    def forward(self, logits, targets):  # compute focal loss
        probs = torch.sigmoid(logits)  # convert logits to probabilities
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)  # probability of correct prediction per element

        if self.alpha is not None:  # if class weights provided
            alpha = self.alpha.view(1, -1)  # shape alpha for broadcasting over batch
            alpha_t = torch.where(targets > 0, alpha, 1.0 - alpha)  # pick alpha for positives and (1-alpha) for negatives
        else:
            alpha_t = 1.0  # no class balancing when alpha is None

        mod_factor = torch.pow(1.0 - pt, self.gamma)  # focal modulating factor
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # element-wise BCE
        loss = alpha_t * mod_factor * ce_loss  # apply alpha and modulating factor

        if self.reduction == "mean":  # mean reduction
            return loss.mean()  # average over batch and classes
        if self.reduction == "sum":  # sum reduction
            return loss.sum()  # sum over all elements
        return loss  # return unreduced loss


class AsymmetricLossMultiLabel(nn.Module):  # asymmetric loss for multi-label problems
    def __init__(self, alpha=None, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, reduction="mean"):  # store parameters
        super().__init__()  # initialize parent nn.Module
        self.alpha = alpha  # per-class positive weights
        self.gamma_pos = gamma_pos  # focusing for positives
        self.gamma_neg = gamma_neg  # focusing for negatives
        self.clip = clip  # optional clipping for negatives
        self.eps = eps  # numerical stability epsilon
        self.reduction = reduction  # reduction mode: mean, sum, or none

        # In the forward pass, compute ASL per element and reduce.
    def forward(self, logits, targets):  # compute asymmetric loss
        probs = torch.sigmoid(logits)  # convert logits to probabilities

        if self.clip is not None and self.clip > 0.0:  # apply optional clipping
            probs_neg = torch.clamp(probs + self.clip, max=1.0)  # clip negative probabilities upward
        else:
            probs_neg = probs  # leave negatives unclipped

        pos_mask = targets  # mask where labels are positive
        neg_mask = 1.0 - targets  # mask where labels are negative

        pos_focus = torch.pow(1.0 - probs, self.gamma_pos)  # focusing term for positives
        neg_focus = torch.pow(probs_neg, self.gamma_neg)  # focusing term for negatives

        log_p_pos = torch.log(probs + self.eps)  # log prob for positives
        log_p_neg = torch.log(1.0 - probs_neg + self.eps)  # log prob for negatives with clipping

        if self.alpha is not None:  # if class weights provided
            alpha = self.alpha.view(1, -1)  # shape alpha for broadcasting
        else:
            alpha = 1.0  # no class weighting if alpha is None

        pos_loss = pos_mask * alpha * pos_focus * log_p_pos  # positive loss term
        neg_loss = neg_mask * neg_focus * log_p_neg  # negative loss term

        loss = -(pos_loss + neg_loss)  # combine and apply negative sign

        if self.reduction == "mean":  # mean reduction
            return loss.mean()  # average over batch and classes
        if self.reduction == "sum":  # sum reduction
            return loss.sum()  # sum over all elements
        return loss  # return unreduced loss
