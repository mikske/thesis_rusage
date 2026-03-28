#%%
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        emb_dim: int = 300,
        num_filters: int = 128,
        kernel_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.5,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_p = dropout
        self.pad_idx = pad_idx

        #слой эмбеддинга
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        #если есть предобученные эмбеддинги, загружаем их
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, emb_dim):
                raise ValueError(
                    "pretrained_embeddings has wrong shape: "
                    f"{tuple(pretrained_embeddings.shape)} "
                    f"!= {(vocab_size, emb_dim)}"
                )

            self.embedding.weight.data.copy_(pretrained_embeddings)

        #при необходимости замораживаем слой эмбеддинга
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        #несколько сверточных веток
        #Conv1d ожидает: (batch, channels, seq_len)
        #channels = emb_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

        #после конкатенации размер будет num_filters * len(kernel_sizes)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        #batch, seq_len, emb_dim
        x = self.embedding(input_ids)

        #batch, emb_dim, seq_len
        x = x.transpose(1, 2)

        pooled_outputs = []

        for conv in self.convs:
            #batch, num_filters, new_seq_len
            conv_out = conv(x)

            #активация
            conv_out = F.relu(conv_out)

            #global max pooling OT
            #batch, num_filters, 1
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))

            #batch, num_filters
            pooled = pooled.squeeze(2)

            pooled_outputs.append(pooled)

        #batch, num_filters * len(kernel_sizes)
        features = torch.cat(pooled_outputs, dim=1)

        features = self.dropout(features)

        logits = self.classifier(features)
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    batch_size = 4
    seq_len = 700
    vocab_size = 30000

    model = TextCNN(
        vocab_size=vocab_size,
        num_classes=2,
        emb_dim=300,
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        dropout=0.5,
        pad_idx=0,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    logits = model(dummy_input)

    print("=== MODEL CHECK ===")
    print(model)
    print("input shape:", dummy_input.shape)
    print("logits shape:", logits.shape)
    print("trainable params:", count_trainable_parameters(model))


if __name__ == "__main__":
    main()