import torch
from tqdm import tqdm
from tabulate import tabulate
from torch.nn import CrossEntropyLoss
from seqeval.metrics import precision_score, recall_score, f1_score

from train_epoch import align_preds
from helpers.settings import DEVICE

@torch.no_grad()
def validate_epoch(model, dataloader, tokenizer, label_list, batches_to_visualize=0):
    model.eval() # make sure model is in eval mode

    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = len(dataloader)
    assert num_batches > 0, "No batches to validate on."
    loss_fn = CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f"Validation set", leave=True) as pbar:
            for batch_idx, batch in enumerate(dataloader):
                pbar.update(1)
                
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits, cache = outputs if isinstance(outputs, tuple) else (outputs, None)
                
                loss = loss_fn(
                    logits.view(-1, model.num_labels),
                    batch["labels"].view(-1),
                )
                total_loss += loss.item()
                pbar.set_postfix(val_loss=total_loss / (pbar.n + 1))

                preds, labels = align_preds(logits, batch["labels"], label_list)
                all_preds.extend(preds)
                all_labels.extend(labels)

                if batch_idx >= batches_to_visualize:
                    continue

                # [x] for development visualization
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                true_labels = batch["labels"].to(DEVICE)
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

                    if running_full_token != '': # flush out the last token
                        rows.append([running_full_token, running_true, running_pred])                    

                    print(tabulate(rows, headers=["Token", "True", "Pred"], tablefmt="pretty"))

    avg_loss = total_loss / num_batches
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"val loss: {avg_loss:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}")

    return {
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }