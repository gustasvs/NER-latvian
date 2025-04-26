from tqdm import tqdm
from torch.nn import CrossEntropyLoss

from helpers.settings import EPOCHS, DEVICE

def train_epoch(epoch_index, model, dataloader, optimizer, scheduler=None):

    model.train() # make sure model is in train mode

    total_loss = 0.0
    num_batches = len(dataloader)
    loss_fn = CrossEntropyLoss(ignore_index=-100)

    with tqdm(total=len(dataloader), desc=f"{epoch_index}/{EPOCHS}", leave=True) as pbar:
        for batch in dataloader:
            pbar.update(1)

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # forward pass
            logits, _ = model(
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
            
            if scheduler: scheduler.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches
    return avg_loss