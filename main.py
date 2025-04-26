import os
from transformers import AutoTokenizer

from mamba import MambaForTokenClassification
from helpers.settings import *
from get_data import get_wikiann_lv
from train_epoch import train_epoch
from validate_epoch import validate_epoch
from helpers.prepare_environment import prepare_environment
from helpers.plot_losses import plot_losses

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
    dropout=0.3
)

# load pre-trained weights if available
# if os.path.exists("models/mamba.pt"):
#     mamba_model.load_state_dict(torch.load("models/mamba.pt"))

mamba_model.to(DEVICE)

optimizer = torch.optim.AdamW(mamba_model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def train():
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        avg_train_loss = train_epoch(
            epoch + 1,
            mamba_model,
            train_dataloader,
            optimizer,
            # scheduler=scheduler,
        )
        avg_val_loss = validate_epoch(
            mamba_model,
            val_dataloader,
            tokenizer,
            label_list,
            batches_to_visualize=0,
        )
        torch.save(
            mamba_model.state_dict(),
            f"models/mamba.pt"
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
    return train_losses, val_losses


if __name__ == "__main__":
    train_losses, val_losses = train()
    plot_losses(train_losses, val_losses)

    





