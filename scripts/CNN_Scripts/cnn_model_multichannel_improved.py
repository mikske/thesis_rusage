from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout1D(nn.Module):
    """
    Dropout по embedding channels.
    На входе: (batch, seq_len, emb_dim)
    """
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        # (batch, seq_len, emb_dim) -> (batch, emb_dim, seq_len)
        x = x.transpose(1, 2)
        x = F.dropout1d(x, p=self.p, training=True)
        # обратно
        x = x.transpose(1, 2)
        return x


class MultiChannelTextCNNImproved(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        emb_dim: int = 300,
        num_filters: int = 128,
        kernel_sizes: List[int] = [3, 4, 5],
        dropout: float = 0.5,
        embedding_dropout: float = 0.2,
        hidden_dim: int = 256,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_static_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_p = dropout
        self.embedding_dropout_p = embedding_dropout
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        # static channel
        self.embedding_static = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        # trainable channel
        self.embedding_trainable = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, emb_dim):
                raise ValueError(
                    "pretrained_embeddings has wrong shape: "
                    f"{tuple(pretrained_embeddings.shape)} != {(vocab_size, emb_dim)}"
                )
            self.embedding_static.weight.data.copy_(pretrained_embeddings)

        if freeze_static_embeddings:
            self.embedding_static.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout1D(embedding_dropout)

        self.static_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )

        self.trainable_convs = nn.ModuleList(
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

        classifier_input_dim = 2 * len(kernel_sizes) * num_filters

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _encode_channel(
        self,
        x: torch.Tensor,
        convs: nn.ModuleList,
    ) -> torch.Tensor:
        # (batch, seq_len, emb_dim) -> regularize embeddings
        x = self.embedding_dropout(x)

        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        pooled_outputs = []

        for conv in convs:
            conv_out = conv(x)
            conv_out = F.relu(conv_out)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled = pooled.squeeze(2)
            pooled_outputs.append(pooled)

        features = torch.cat(pooled_outputs, dim=1)
        return features

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        static_x = self.embedding_static(input_ids)
        static_features = self._encode_channel(static_x, self.static_convs)

        trainable_x = self.embedding_trainable(input_ids)
        trainable_features = self._encode_channel(trainable_x, self.trainable_convs)

        features = torch.cat([static_features, trainable_features], dim=1)
        features = self.dropout(features)

        logits = self.classifier(features)
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    batch_size = 4
    seq_len = 700
    vocab_size = 30000
    emb_dim = 300

    pretrained_embeddings = torch.randn(vocab_size, emb_dim)

    model = MultiChannelTextCNNImproved(
        vocab_size=vocab_size,
        num_classes=2,
        emb_dim=emb_dim,
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        dropout=0.5,
        embedding_dropout=0.2,
        hidden_dim=256,
        pad_idx=0,
        pretrained_embeddings=pretrained_embeddings,
        freeze_static_embeddings=True,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    logits = model(dummy_input)

    print("=== MODEL CHECK ===")
    print(model)
    print("input shape:", dummy_input.shape)
    print("logits shape:", logits.shape)
    print("trainable params:", count_trainable_parameters(model))
    print("static requires_grad:", model.embedding_static.weight.requires_grad)
    print("trainable requires_grad:", model.embedding_trainable.weight.requires_grad)