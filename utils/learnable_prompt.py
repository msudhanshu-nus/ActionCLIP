import torch
import torch.nn as nn
import clip

class LearnablePromptEncoder(nn.Module):
    def __init__(self, clip_model, prefix=2, postfix=2, std_init=0.02):
        super().__init__()
        self.clip_model = clip_model
        self.prefix = prefix
        self.postfix = postfix
        width = clip_model.token_embedding.weight.shape[1]
        self.prefix_embed = nn.Parameter(torch.randn(prefix, width) * std_init)
        self.postfix_embed = nn.Parameter(torch.randn(postfix, width) * std_init)
        self.dtype = clip_model.dtype  # fp16 if model was converted

    def forward(self, texts):
        # texts: list of strings
        tokens = clip.tokenize(texts, context_length=self.max_tokens).to(next(self.parameters()).device)
        # start with CLIP token embeddings
        x = self.clip_model.token_embedding(tokens).type(self.dtype)  # [B, 77, D]
        B, L, D = x.shape
        # find end-of-text index per sample
        eot = tokens.argmax(dim=-1)
        # insert learnable prefix just after the BOS (pos 1..prefix)
        x[:, 1:1+self.prefix] = self.prefix_embed.unsqueeze(0).expand(B, -1, -1)
        # insert learnable postfix right before the EOT token slot
        for i in range(B):
            end = eot[i].item()
            start_post = min(end, 1 + self.prefix + (end - 1 - self.prefix - self.postfix))
            # fill postfix; if truncated, it just overwrites the tokens before EOT
            x[i, end - self.postfix:end] = self.postfix_embed
        # CLIP text encoder (copied from openai-clip encode_text)
        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.dtype)
        x = x[torch.arange(B), eot] @ self.clip_model.text_projection
        return x