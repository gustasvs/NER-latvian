
import datasets
from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from tabulate import tabulate

from transformers import AutoTokenizer

from helpers.settings import *

def tokenize_and_align_labels(example, tokenizer):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SAMPLE_LENGTH,
        padding="max_length",
    )
    word_ids = tokenized.word_ids()
    
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["ner_tags"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized

def preview_dataloader(dataloader, tokenizer, label_list, num_samples=1):
    """Prints a human-readable preview of the first batch."""
    batch = next(iter(dataloader))  # fetch first batch

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    for sample_idx in range(num_samples):
        input_id = input_ids[sample_idx]
        mask = attention_mask[sample_idx]
        label = labels[sample_idx]

        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_id.tolist())

        # Build table rows
        table = []
        for tok, m, lab in zip(tokens, mask, label):
            if m.item() == 1:  # Only non-padded tokens
                label_name = label_list[lab] if lab >= 0 else "-"
                table.append([tok, label_name])

        print(f"\nSample {sample_idx + 1}")
        print(tabulate(table, headers=["Token", "NER Label"], tablefmt="pretty"))

def get_wikiann_lv(tokenizer):
    """Loads the WikiANN-lv dataset and returns train/val/test (pytorch) DataLoaders / (tensorflow) datasets."""
    
    ds = load_dataset("wikiann", "lv")
    
    if MAX_SAMPLES_TO_USE != -1:
        ds["train"] = ds["train"].shuffle(seed=42).select(range(MAX_SAMPLES_TO_USE))
        ds["validation"] = ds["validation"].shuffle(seed=42).select(range(MAX_SAMPLES_TO_USE))
        ds["test"] = ds["test"].shuffle(seed=42).select(range(MAX_SAMPLES_TO_USE))

    fn = partial(tokenize_and_align_labels, tokenizer=tokenizer)
    ds = ds.map(fn, batched=False)

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    label_list = ds["train"].features["ner_tags"].feature.names
    num_labels = len(label_list)


    train_loader = DataLoader(ds["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ds["validation"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(ds["test"], batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, num_labels, label_list


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    train_loader, val_loader, test_loader, num_labels, label_list = get_wikiann_lv(tokenizer)
    print(f"Number of labels: {num_labels}")
    print(f"Label list: {label_list}")
    print(f"Train loader size: {len(train_loader)}")
    print(f"Validation loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}")

    preview_dataloader(train_loader, tokenizer, label_list, num_samples=2)
