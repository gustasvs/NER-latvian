import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from seqeval.metrics import precision_score, recall_score, f1_score

from helpers.settings import EPOCHS, DEVICE

def align_preds(logits, label_ids, label_list):

    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    label_ids = label_ids.cpu().numpy()

    true_preds, true_labels = [], []
    for p_row, l_row in zip(preds, label_ids):
        sent_pred, sent_label = [], []
        for p, l in zip(p_row, l_row):
            # ignore index
            if l == -100:
                continue
            sent_pred.append(label_list[p])
            sent_label.append(label_list[l])
        true_preds.append(sent_pred)
        true_labels.append(sent_label)
    return true_preds, true_labels

def train_epoch(epoch_index, model, dataloader, optimizer, label_list):

    model.train() # make sure model is in train mode

    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = len(dataloader)
    loss_fn = CrossEntropyLoss(ignore_index=-100)


    with tqdm(total=len(dataloader), desc=f"{epoch_index}/{EPOCHS}", leave=True) as pbar:
        for batch in dataloader:
            # tqdm.write(f"Batch {pbar.n + 1}/{num_batches}", end='\r')
            # tqdm.write(batch)
            pbar.update(1)

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # forward pass
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = loss_fn(
                logits.view(-1, model.num_labels),
                batch["labels"].view(-1),
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if scheduler: scheduler.step()

            total_loss += loss.item()

            # metrics
            # preds, labels = align_preds(logits.detach(), batch["labels"], label_list)
            # all_preds.extend(preds)
            # all_labels.extend(labels)
            # f1_running = f1_score(all_labels, all_preds) if all_preds else 0.0
            # f1_running = 0.0

            if (pbar.n + 1) % 10 == 0:
                pbar.set_postfix(loss=total_loss / (pbar.n + 1), memo_aloc=torch.cuda.memory_allocated(), memo_reserved=torch.cuda.memory_reserved())

    avg_loss = total_loss / num_batches
    # precision = precision_score(all_labels, all_preds)
    # recall = recall_score(all_labels, all_preds)
    # f1 = f1_score(all_labels, all_preds)
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    return {
        "loss": avg_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }