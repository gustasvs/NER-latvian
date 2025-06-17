import os
import numpy as np
from tqdm import tqdm
import pickle, json

from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset, ClassLabel, Sequence
from functools import partial
from torch.utils.data import DataLoader
from tabulate import tabulate

from transformers import AutoTokenizer, DataCollatorForTokenClassification

from helpers.settings import *

tag_map = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-person": 1,  # alias for B-PER
    "I-person": 2,  # alias for I-PER
    "B-ORG": 3,
    "I-ORG": 4,
    "B-organization": 3,  # alias for B-ORG
    "I-organization": 4,  # alias for I-ORG
    "B-LOC": 5,
    "I-LOC": 6,
    "B-location": 5,  # alias for B-LOC
    "I-location": 6,  # alias for I-LOC
    "B-MISC": 7, # all the other tags like entity, GPE, etc.
    "I-MISC": 8,
}

label_list = [
    "O", 
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]

def tokenize_and_align_labels(sentence, tokenizer):
    # sentence["tokens"] = [['Nelly', '(', '3', ')'], ['Kurts', 'Aterbergs', '(', "''Kurt", 'Magnus', 'Atterberg', "''", ',', '1887—1974', ')', '—', 'komponists', ';'], ['Igors', 'Stepanovs', '(', '10', '.'], ['1961.—1962.', 'gada', 'NBA', 'sezona'], ['sens', 'nosaukums', '(', 'Ptolemajs', ')']]
    # sentence["ner_tags"] =  [[1, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0], [3, 4, 4, 4], [0, 0, 0, 1, 0]]
    words = sentence["tokens"]
    ner_tags = sentence["ner_tags"]
    
    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SAMPLE_LENGTH,
        padding="max_length",
    )
    word_ids = tokenized.word_ids()

    if len(tokenized['input_ids']) != 64:
        raise ValueError(f"Tokenization error: {len(tokenized['input_ids'])} != 64. Check the input sentence: {sentence['tokens']}")
    
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(ner_tags[word_idx])
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
        headers = ["Token"] + [row[0] for row in table]
        rows = [["NER Label"] + [row[1] for row in table]]
        print(tabulate(rows, headers=headers, tablefmt="pretty"))

def rebalance_splits(ds, n_move=8000):
    val_ds, test_ds = ds["validation"], ds["test"]
    val_len, test_len = len(val_ds), len(test_ds)

    val_to_train  = val_ds.select(range(val_len - n_move, val_len))
    test_to_train = test_ds.select(range(test_len - n_move, test_len))

    new_train      = concatenate_datasets([ds["train"], val_to_train, test_to_train])
    new_validation = val_ds.select(range(0, val_len - n_move))
    new_test       = test_ds.select(range(0, test_len - n_move))

    return DatasetDict({
      "train":      new_train,
      "validation": new_validation,
      "test":       new_test
    })

def ner_tag_map(tag):
    # https://huggingface.co/datasets/SEACrowd/wikiann
    # O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4), B-LOC (5), I-LOC (6)
    
    # if not found print it and stop the application
    is_unknown = tag not in tag_map
    is_start = tag.startswith("B")
    if is_unknown:
        # print(f"Unknown NER tag: {tag}")
        if is_start:
            # B-MISC
            return 7
        else:
            # I-MISC
            return 8

    return tag_map.get(tag, 7)

def extract_sentences_from_conll(file):
    sentences = []
    current_sentence = {"words": [], "ner_tags": [], 'id': None}

    # first 2 lines are metadata
    lines = file.readlines()[2:]

    for line in lines:
        line = line.strip()
        # empty lines divide sentences
        if not line:
            if current_sentence["words"]:
                sentences.append(current_sentence)
                current_sentence = {"words": [], "ner_tags": []}
            continue

        # first two lines of the sentences are also a sentence specific metadata
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                current_sentence["id"] = line.split(" = ")[1].strip()
            continue

        line_parts = line.split("\t")
        word = line_parts[1]
        ner_tag = line_parts[6]

        current_sentence["words"].append(word)
        current_sentence["ner_tags"].append(ner_tag_map(ner_tag))

    # there should be an empty line at the end of the file, but just in case
    if current_sentence["words"]:
        sentences.append(current_sentence)

    return sentences


def get_LUMII_lv():
    """Loads the LUMII-lv dataset and returns array of data."""
    
    if not os.path.exists("data/LUMII-AiLab"):
        raise FileNotFoundError("LUMII-AiLab dataset not found. Fetch it using `fetch_LUMII_data.py` script.")
    
    data = []
    sentences_with_atleast_one_entity = 0

    files = os.listdir("data/LUMII-AiLab")

    with tqdm(files, desc="Processing LUMII files") as pbar:
        for conll_file in pbar:
            with open(os.path.join("data/LUMII-AiLab", conll_file), "r", encoding="utf-8") as f:
                sentences_in_file = extract_sentences_from_conll(f)
                data.extend(sentences_in_file)

                for sentence in sentences_in_file:
                    if any(tag != 0 for tag in sentence["ner_tags"]):
                        sentences_with_atleast_one_entity += 1

            pbar.set_postfix_str(
                f"Total: {str(len(data)).ljust(5)}, With atleast one NER tag: {str(sentences_with_atleast_one_entity).ljust(5)}"
            )
            # tqdm.write(f"sentences in file: {conll_file} -> {sentences_in_file}")
    return Dataset.from_dict({
        "tokens": [sentence["words"] for sentence in data],
        "ner_tags": [sentence["ner_tags"] for sentence in data],
        # "id": [sentence["id"] for sentence in data]
    })


def get_wikiann_lv():
    """Loads the WikiANN-lv dataset and returns train/val/test (pytorch) DataLoaders / (tensorflow) datasets."""
    
    ds = load_dataset("wikiann", "lv")

    # print(ds["train"][:5])

    ds = rebalance_splits(ds, 7000)

    print(f"Dataset size: {len(ds['train'])} train | {len(ds['validation'])} val | {len(ds['test'])} test")
    
    if MAX_SAMPLES_TO_USE != -1:
        ds["train"] = ds["train"].shuffle(seed=42).select(range(MAX_SAMPLES_TO_USE))
        ds["validation"] = ds["validation"].shuffle(seed=42).select(range(MAX_SAMPLES_TO_USE))
        ds["test"] = ds["test"].shuffle(seed=42).select(range(MAX_SAMPLES_TO_USE))

    return ds

def get_translated_wikiann_lv():
    if not os.path.exists("translated"):
        raise FileNotFoundError("Translated WikiANN dataset not found. Generate it by running `translation.py` script.")

    # with open("translated_v1/conll_train_pure.pkl", "rb") as f:
    #     train_data = pickle.load(f)

    # with open("translated_v1/conll_validation_pure.pkl", "rb") as f:
    #     val_data = pickle.load(f)

    # with open("translated_v1/conll_test_pure.pkl", "rb") as f:
    #     test_data = pickle.load(f)

    with open("translated_ner_fixed/fixed_ner_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open("translated_ner_fixed/fixed_ner_validation.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open("translated_ner_fixed/fixed_ner_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    def string_to_word_list(s):
        return s.split()

    ds = DatasetDict({
        "train": Dataset.from_dict({
            "tokens": [string_to_word_list(item[0]) for item in train_data],
            "ner_tags": [[ner_tag_map(tag) for tag in item[1]] for item in train_data],
        }),
        "validation": Dataset.from_dict({
            "tokens": [string_to_word_list(item[0]) for item in val_data],
            "ner_tags": [[ner_tag_map(tag) for tag in item[1]] for item in val_data],
        }),
        "test": Dataset.from_dict({
            "tokens": [string_to_word_list(item[0]) for item in test_data],
            "ner_tags": [[ner_tag_map(tag) for tag in item[1]] for item in test_data],
        })
    })
    return ds


def get_data_loaders(tokenizer, wikiann=True, lumii=True, translated_wikiann=True, use_cache=True):
    """
        val dataloader ALWAYS consists of the combination of WikiANN and LUMII datasets.
        returns train, val, test dataloaders
    """

    if not wikiann and not lumii and not translated_wikiann:
        raise ValueError("get_data_loaders function expects at least one dataset to be True.")

    fn = partial(tokenize_and_align_labels, tokenizer=tokenizer)
    combined_parts = {"train": [], "validation": [], "test": []}

    if translated_wikiann:
        if use_cache and os.path.exists("cache/translated_wikiann.pkl"):
            print("Loading Translated WikiANN dataset from cache...")
            with open("cache/translated_wikiann.pkl", "rb") as f:
                translated_wikiann_ds = pickle.load(f)
        else:
            translated_wikiann_ds = get_translated_wikiann_lv()
            print("Tokenizing Translated WikiANN dataset...")
            translated_wikiann_ds = translated_wikiann_ds.map(fn, batched=False)
            # same way here, we have to add the MISC labels manually
            translated_wikiann_ds = translated_wikiann_ds.cast_column(
                "ner_tags",
                Sequence(feature=ClassLabel(names=label_list))
            )
            with open("cache/translated_wikiann.pkl", "wb") as f:
                pickle.dump(translated_wikiann_ds, f)

        # we never use translated data for validation or testing
        combined_parts["train"].extend([
            translated_wikiann_ds["train"],
            translated_wikiann_ds["validation"],
            translated_wikiann_ds["test"]
        ])


    # if wikiann:
    # 
    if use_cache and os.path.exists("cache/wikiann.pkl"):
        print("Loading WikiANN dataset from cache...")
        with open("cache/wikiann.pkl", "rb") as f:
            wikiann_ds = pickle.load(f)

    else:
        wikiann_ds = get_wikiann_lv()
        print("Tokenizing WikiANN dataset...")
        wikiann_ds = wikiann_ds.map(fn, batched=False)
        # since it doesnt have the MISC labels we have to add them manually
        wikiann_ds = wikiann_ds.cast_column(
            "ner_tags",
            Sequence(feature=ClassLabel(names=label_list))
        )
        with open("cache/wikiann.pkl", "wb") as f:
            pickle.dump(wikiann_ds, f)
    
    if wikiann:
        combined_parts["train"].append(wikiann_ds["train"])
    # wikiann always goes to validation and test splits
    combined_parts["validation"].append(wikiann_ds["validation"])
    combined_parts["test"].append(wikiann_ds["test"])


    if use_cache and os.path.exists("cache/lumii.pkl"):
        print("Loading LUMII dataset from cache...")
        with open("cache/lumii.pkl", "rb") as f:
            lumii_ds = pickle.load(f)

            lumii_train = lumii_ds["train"]
            lumii_val = lumii_ds["validation"]
            lumii_test = lumii_ds["test"]
    else: 
        lumii_ds = get_LUMII_lv()
        print("Tokenizing LUMII dataset...")
        lumii_ds = lumii_ds.map(fn, batched=False)
        # same way here, we have to add the MISC labels manually
        lumii_ds = lumii_ds.cast_column(
            "ner_tags",
            Sequence(feature=ClassLabel(names=label_list))
        )

        n = len(lumii_ds)
        t_sz, v_sz = int(0.1 * n), int(0.1 * n)

        lumii_test = lumii_ds.select(range(t_sz))
        lumii_val = lumii_ds.select(range(t_sz, t_sz + v_sz))
        lumii_train = lumii_ds.select(range(t_sz + v_sz, n))

        with open("cache/lumii.pkl", "wb") as f:
            pickle.dump({
                "train": lumii_train,
                "validation": lumii_val,
                "test": lumii_test
            }, f)

    if lumii:
        combined_parts["train"].append(lumii_train)

    # lumii always goes to validation and test splits
    combined_parts["validation"].append(lumii_val)
    combined_parts["test"].append(lumii_test)

    combined_ds = DatasetDict({
        split: concatenate_datasets(splits) if len(splits) > 1 else splits[0]
        for split, splits in combined_parts.items()
    })
    # clean up columns
    combined_ds = combined_ds.remove_columns(["tokens", "ner_tags", 'langs', 'spans'])
    
    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    train_loader = DataLoader(combined_ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(combined_ds["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator)
    test_loader = DataLoader(combined_ds["test"], batch_size=BATCH_SIZE, collate_fn=data_collator)

    return train_loader, val_loader, test_loader, len(label_list), label_list

if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased", resume_download=None)

    train_loader, val_loader, test_loader, num_labels, label_list = get_data_loaders(tokenizer, wikiann=True, lumii=True)


    print(f"Number of labels: {num_labels}")
    print(f"Label list: {label_list}")
    print(f"Train loader size: {len(train_loader)}, total samples: {len(train_loader.dataset)}")
    print(f"Validation loader size: {len(val_loader)}, total samples: {len(val_loader.dataset)}")
    print(f"Test loader size: {len(test_loader)}, total samples: {len(test_loader.dataset)}")

    # preview_dataloader(train_loader, tokenizer, label_list, num_samples=20)
