from mamba import MambaForTokenClassification
import os
from transformers import AutoTokenizer

from mamba import MambaForTokenClassification
from helpers.settings import *

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

from helpers.settings import *


def count_params(num_layers, d_model, d_state):
    m = MambaForTokenClassification(
        vocab_size=tokenizer.vocab_size,
        num_labels=7,
        num_layers=num_layers,
        d_input=MAX_SAMPLE_LENGTH,
        d_model=d_model,
        d_state=d_state,
        d_discr=None,
        ker_size=4,
        parallel=False,
        dropout=0.5,
        bi_directional=True,
    )
    return sum(p.numel() for p in m.parameters()) / 1e6


# Example sweep
for L in [6, 8, 24, 48]:
    for D in [256, 384, 512, 1024]:
        for S in [64, 128, 256]:
            sz = count_params(L, D, S)
            if any(abs(sz-t)<5 for t in [33, 67, 134,336]):
                print(f"L={L}, d_model={D}, d_state={S} → {sz:.1f}M")
