import torch.nn.functional as F
import torch.nn as nn

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
