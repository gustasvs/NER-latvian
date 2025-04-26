import torch
from tqdm import tqdm
from tabulate import tabulate

from helpers.settings import EPOCHS, DEVICE


def validate_epoch(model, dataloader, tokenizer, label_list, max_batches=None):
    model.eval() # make sure model is in eval mode
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            true_labels = batch["labels"].to(DEVICE)

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1)

            for i in range(input_ids.size(0)):
                # Skip padding tokens
                mask = attention_mask[i].bool()
                ids = input_ids[i][mask]
                tokens = tokenizer.convert_ids_to_tokens(ids.tolist())

                text = tokenizer.decode(ids, skip_special_tokens=True)
                print(f"\nSentence: {text}")

                rows = []
                running_full_token = ''
                running_true = ''
                running_pred = ''
                for tok, t_id, p_id in zip(tokens,
                                           true_labels[i][mask].tolist(),
                                           preds[i][mask].tolist()):
                    if t_id < 0: # skip sub-tokens
                        running_full_token += tok.replace('##', '')
                        continue
                    
                    true_tag = label_list[t_id] if t_id >= 0 else '-'
                    pred_tag = label_list[p_id]

                    # if running_full_token:
                        # flush out the previous token
                    rows.append([running_full_token, running_true, running_pred])
                    running_full_token = tok.replace('##', '')
                    running_true = true_tag
                    running_pred = pred_tag
                    

                print(tabulate(rows, headers=["Token", "True", "Pred"], tablefmt="pretty"))

            if max_batches and batch_idx + 1 >= max_batches:
                break