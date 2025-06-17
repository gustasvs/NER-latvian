import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, AutoModel

class DistilBertForTokenClassification(nn.Module):
    def __init__(
        self,
        num_labels: int,
        pretrained_model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()
        # backbone
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        # self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        hidden = self.bert.config.hidden_size

        # classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
    ):
        # (B, L, H)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)  # (B, L, num_labels)

        if labels is not None:
            # Flatten tokens for CE (ignore padding token labels = -100)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )
            return loss, logits

        return logits
