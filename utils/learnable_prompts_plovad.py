import torch
import torch.nn as nn
import clip

class PlovadLearnablePrompt(nn.Module):
    def __init__(self, clip_model, prefix=2, postfix=2, std_init=0.02, max_tokens=77):
        super().__init__()
        self.clip_model = clip_model                     # hold the frozen CLIP model
        self.prefix = prefix                             # number of learnable tokens before the text
        self.postfix = postfix                           # number of learnable tokens after the text
        self.max_tokens = max_tokens                     # CLIP’s max token length (77)
        width = clip_model.token_embedding.weight.shape[1]  # embedding width (e.g., 512 for ViT-B/16)
        # trainable embedding table for all positions (mirrors PLOVAD’s nn.Embedding(77, 512))
        self.embedding = nn.Embedding(max_tokens, width)
        nn.init.normal_(self.embedding.weight, std=std_init)  # init like PLOVAD’s std_init
        self.dtype = clip_model.dtype                    # keep dtype consistent (fp16 if converted)

    def forward(self, texts):
        # texts: list of strings
        word_tokens = clip.tokenize(texts, context_length=self.max_tokens).to(next(self.parameters()).device)  # [B, 77]
        # CLIP’s token embeddings for the raw text
        word_embedding = self.clip_model.encode_token(word_tokens)                            # [B, 77, D]
        # learnable prompt embeddings for all token slots, expanded per batch
        text_embeddings = self.embedding(torch.arange(self.max_tokens, device=word_tokens.device))
        text_embeddings = text_embeddings.unsqueeze(0).expand(len(texts), -1, -1).type(self.dtype)  # [B, 77, D]
        # placeholder for token ids (needed by CLIP’s text encoder)
        text_tokens = torch.zeros_like(word_tokens)                                          # [B, 77]

        # PLOVAD-style insertion of prefix/postfix around the original tokens
        for i in range(len(texts)):
            ind = torch.argmax(word_tokens[i], -1).item()  # position of EOT token for this sample
            # copy CLS/BOS token from the original embedding
            text_embeddings[i, 0] = word_embedding[i, 0]
            # copy original text tokens into the learnable buffer with a prefix shift
            text_embeddings[i, self.prefix + 1: self.prefix + ind] = word_embedding[i, 1: ind]
            # copy the EOT token into the shifted position after postfix
            text_embeddings[i, self.prefix + ind + self.postfix] = word_embedding[i, ind]

            # mirror the token IDs into text_tokens (PLOVAD keeps IDs aligned)
            text_tokens[i, 0] = word_tokens[i, 0]                                           # BOS
            text_tokens[i, self.prefix + 1: self.prefix + ind] = word_tokens[i, 1: ind]     # shifted body
            text_tokens[i, self.prefix + ind + self.postfix] = word_tokens[i, ind]          # shifted EOT

        # run CLIP’s text encoder on the modified embeddings/tokens
        text_features = self.clip_model.encode_text(text_embeddings, text_tokens)            # [B, D]
        return text_features