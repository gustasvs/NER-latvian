import torch
import torch.nn as nn
from mamba_backbone import Mamba

class MambaForTokenClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        num_layers: int,
        d_input: int,
        d_model: int,
        d_state: int = 16,
        d_discr: int | None = None,
        ker_size: int = 4,
        parallel: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        # embeddings
        self.embeddings = nn.Embedding(vocab_size, d_input)
        # encoder
        self.encoder = Mamba(
            num_layers=num_layers,
            d_input=d_input,
            d_model=d_model,
            d_state=d_state,
            d_discr=d_discr,
            ker_size=ker_size,
            parallel=parallel
        )
        # classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_input, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        cache: tuple = None,
    ):
        # Embed tokens → (B, L, d_input)
        x = self.embeddings(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        
        # Encode sequence → (B, L, d_input)
        x, new_cache = self.encoder(x, cache=cache)
        
        # Head → logits (B, L, num_labels)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits, new_cache
