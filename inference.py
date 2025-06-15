import torch
from transformers import AutoTokenizer
from mamba import MambaForTokenClassification
from helpers.settings import MAX_SAMPLE_LENGTH, DEVICE
from get_data import get_wikiann_lv

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    _, _, _, num_labels, label_list = get_wikiann_lv(tokenizer)

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
    checkpoint = torch.load("models/mamba.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()

    while True:

        text = input("Enter a sentence: ")
        if text == "q": break
        
        words = text.split()

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_SAMPLE_LENGTH,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1).squeeze().tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        mask   = attention_mask.squeeze().tolist()
        print("-----------------------------")
        words, labels = [], []

        for tok, m, p in zip(tokens, mask, preds):
            if m == 1 and tok not in tokenizer.all_special_tokens:
                if tok.startswith('##'):
                    words[-1] += tok[2:]
                else:
                    words.append(tok)
                    labels.append(label_list[p])

        line_labels = " ".join(lbl.center(max(len(w), len(lbl) )) for lbl, w in zip(labels, words))
        line_words  = " ".join(w.center(max(len(w), len(lbl) )) for lbl, w in zip(labels, words))

        print(line_labels)
        print(line_words)


if __name__ == "__main__":
    main()
