from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
#мулуьти-канальная нейронная сеть
class MultiChannelCNN(nn.Module):
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
            freeze_static_embeddings: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_sizes
        self.dropout_p = dropout
        self.pad_idx = pad_idx

        #статичный канал fastText embeddings
        self.embedding_static = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        #обычный обучающий слой
        self.embedding_trainable = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx,
        )

        #загружаем предобученные эмбеддинги в статичный канал
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, emb_dim):
                raise ValueError(
                    "pretrained_embeddings has wrong shape: "
                    f"{tuple(pretrained_embeddings.shape)} "
                    f"!= {(vocab_size, emb_dim)}"
                )

            self.embedding_static.weight.data.copy_(pretrained_embeddings)

        #замораживаем статичный канал, чтобы он оставался стабильным и хранил знания fasttext
        if freeze_static_embeddings:
            self.embedding_static.weight.requires_grad = False

        #сверточные ветки для статичного канала
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

        #сверточные ветки для обучаемого канала
        #вообще можно было сделать общие слои для двух каналов, но тут бы хоть так запустилось
        self.trainable_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters,
                    kernel_size=k
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

        #у нас 2 канала * len(kernel_sizes) * num_filters
        classifier_input_dim = 2 * len(kernel_sizes) * num_filters

        self.classifier = nn.Linear(classifier_input_dim, num_classes)

        #вспомогательная функция для одного канала
        #х: (batch, seq_len, emb_dim)
        #returns: (batch, len(kernel_sizes) * num_filters)
        #conv1d: (batch, channels, seq_len)
    def _encode_channel(
            self,
            x: torch.Tensor,
            convs: nn.ModuleList,
    ) -> torch.Tensor:
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

    #input_ids: (batch_size, seq_len)
    #logits: (batch_size, num_classes)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        #статический канал
        static_x = self.embedding_static(input_ids)
        static_features = self._encode_channel(static_x, self.static_convs)

        #обучаемый канал
        trainable_x = self.embedding_trainable(input_ids)
        trainable_features = self._encode_channel(trainable_x, self.trainable_convs)

        #конкатенация обоих слоев
        features = torch.cat([static_features, trainable_features], dim=1)

        features = self.dropout(features)

        logits = self.classifier(features)
        return logits

    #подсчет обучаемых параметров
def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main() -> None:
    batch_size = 4
    seq_len = 700
    vocab_size = 30000
    emb_dim = 300

    pretrained_embeddings = torch.randn(vocab_size, emb_dim)

    model = MultiChannelCNN(
        vocab_size=vocab_size,
        num_classes=2,
        emb_dim=emb_dim,
        num_filters=128,
        kernel_sizes=[3, 4, 5],
        dropout=0.5,
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
    print("classifier out_features:", model.classifier.out_features)

if __name__ == "__main__":
    main()