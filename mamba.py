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
        dropout: float = 0.1,
        bi_directional: bool = False,
    ):
        super().__init__()
        # embeddings
        self.embeddings = nn.Embedding(vocab_size, d_input)
        
        # encoder
        self.encoder_fwd = Mamba(
            num_layers=num_layers,
            d_input=d_input,
            d_model=d_model,
            d_state=d_state,
            d_discr=d_discr,
            ker_size=ker_size,
            parallel=parallel
        )
        self.encoder_bwd = Mamba(
            num_layers=num_layers,
            d_input=d_input,
            d_model=d_model,
            d_state=d_state,
            d_discr=d_discr,
            ker_size=ker_size,
            parallel=parallel
        ) if bi_directional else None

        # classification head
        self.dropout = nn.Dropout(dropout)
        _d_input = d_input * 2 if bi_directional else d_input
        self.classifier = nn.Linear(_d_input, num_labels)
        self.num_labels = num_labels
        self.bi_directional = bi_directional

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
        x, new_cache = self.encoder_fwd(x, cache=cache)
        
        if self.bi_directional:
            # Encode sequence in reverse direction → (B, L, d_input)
            x_rev = torch.flip(x, dims=[1])
            x_bwd, _ = self.encoder_bwd(x_rev, cache=cache)
            x = torch.cat([x, torch.flip(x_bwd, dims=[1])], dim=-1)


        # Head → logits (B, L, num_labels)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits, new_cache
