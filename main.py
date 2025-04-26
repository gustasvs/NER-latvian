import os
from transformers import AutoTokenizer

from mamba import MambaForTokenClassification
from helpers.settings import *
from get_data import get_wikiann_lv
from train_epoch import train_epoch
from validate_epoch import validate_epoch
from helpers.prepare_environment import prepare_environment

prepare_environment(os.path.dirname(os.path.abspath(__file__)))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

train_dataloader, val_dataloader, test_dataloader, num_labels, label_list = get_wikiann_lv(tokenizer)

mamba_model = MambaForTokenClassification(
    vocab_size=tokenizer.vocab_size,
    num_labels=num_labels,
    num_layers=4,
    d_input=MAX_SAMPLE_LENGTH,
    d_model=512,
    d_state=16,
    d_discr=None,
    ker_size=4,
    parallel=False,
    dropout=0.1
)

# load pre-trained weights if available
if os.path.exists("models/mamba.pt"):
    mamba_model.load_state_dict(torch.load("models/mamba.pt"))


mamba_model.eval()
mamba_model.to(DEVICE)

optimizer = torch.optim.AdamW(mamba_model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

for epoch in range(EPOCHS):
    train_epoch(
        epoch + 1,
        mamba_model,
        train_dataloader,
        optimizer,
        # scheduler=scheduler,
    )
    validate_epoch(
        mamba_model,
        val_dataloader,
        tokenizer,
        label_list,
        1
    )
    torch.save(
        mamba_model.state_dict(),
        f"models/mamba.pt"
    )

    





