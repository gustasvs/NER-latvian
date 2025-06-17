import os
from transformers import AutoTokenizer

from mamba import MambaForTokenClassification
from transformer import DistilBertForTokenClassification
from helpers.settings import *
from get_data import get_data_loaders
from train_epoch import train_epoch
from validate_epoch import validate_epoch
from helpers.prepare_environment import prepare_environment
from helpers.plot_losses import plot_losses

prepare_environment(os.path.dirname(os.path.abspath(__file__)))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")


def get_model(name, num_labels):
    if name == "mamba":
        model = MambaForTokenClassification(
            vocab_size=tokenizer.vocab_size,
            num_labels=num_labels,
            num_layers=4,
            d_input=MAX_SAMPLE_LENGTH,
            d_model=256,
            d_state=16,
            d_discr=None,
            ker_size=4,
            parallel=False,
            dropout=0.5,
            bi_directional=True,
        )
    elif name == "distilbert":
        model = DistilBertForTokenClassification(
            num_labels=num_labels,
            pretrained_model_name="distilbert-base-multilingual-cased",
            dropout=0.1,
        )

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model.to(DEVICE)
    print(f"Model is on {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return model, optimizer  # , scheduler

def train(model, optimizer, train_dataloader, val_dataloader, label_list):
    # store final outputs
    training_metrics = []
    validation_metrics = []
    
    # for early stopping
    lowest_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_metrics = train_epoch(
            epoch + 1,
            model,
            train_dataloader,
            optimizer,
            label_list,
        )
        val_metrics = validate_epoch(
            model,
            val_dataloader,
            tokenizer,
            label_list,
            batches_to_visualize=0,
        )

        training_metrics.append(train_metrics)
        validation_metrics.append(val_metrics)

        if val_metrics["loss"] < lowest_val_loss:
            lowest_val_loss = val_metrics["loss"]
            
            # save only the best model
            torch.save(
                model.state_dict(),
                f"models/mamba_best.pt"
            )
        else:
            print(f"Validation loss {val_metrics['loss']:.2f} did not improve from {lowest_val_loss:.2f}, stopping training.")
            # early stopping
            break
            
    
    return training_metrics, validation_metrics


def main():
    # performs grid search for models and data.
    # models = ["mamba", "distilbert"]
    models = ["distilbert"]
    datasets = ["wikiann", "lumii", "translated_wikiann"]

    for model in models:
        for dataset in datasets:
            print(f"{'-' * 5} Training {model} on {dataset} {'-' * 5}")
            train_dataloader, val_dataloader, test_dataloader, num_labels, label_list = \
                get_data_loaders(tokenizer, True, True, True, True)
                # get_data_loaders(tokenizer, dataset, w

            model, optimizer = get_model(model, num_labels)

            train_metrics, val_metrics = train(
                model,
                optimizer,
                train_dataloader,
                val_dataloader,
                label_list,
            )

            print(f"training_metrics: {train_metrics}")
            print(f"validation_metrics: {val_metrics}")


if __name__ == "__main__":
    main()
