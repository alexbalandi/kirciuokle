"""Torch models and collators for the configurable tagger heads."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput

from head_config import slot_output_name


LABEL_PAD_ID = -100


def hidden_size_from_config(config: object) -> int:
    for name in ("hidden_size", "d_model", "dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"could not determine hidden size from {config!r}")


def encoder_kwargs(
    encoder: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_type_ids: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    parameters = inspect.signature(encoder.forward).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, torch.Tensor] = {"input_ids": input_ids}
    if attention_mask is not None and ("attention_mask" in parameters or accepts_kwargs):
        kwargs["attention_mask"] = attention_mask
    if token_type_ids is not None and ("token_type_ids" in parameters or accepts_kwargs):
        kwargs["token_type_ids"] = token_type_ids
    return kwargs


@dataclass
class PooledDataCollator:
    tokenizer: object
    slot_count: int = 0
    label_pad_token_id: int = LABEL_PAD_ID

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        custom_keys = {
            "labels",
            "slot_labels",
            "first_subword_indices",
            "last_subword_indices",
        }
        base_features = [
            {key: value for key, value in feature.items() if key not in custom_keys}
            for feature in features
        ]
        batch = self.tokenizer.pad(base_features, padding=True, return_tensors="pt")

        labels = [feature.get("labels", []) for feature in features]
        if labels:
            batch["labels"] = self._pad_1d(labels, self.label_pad_token_id)

        slot_labels = [feature.get("slot_labels") for feature in features]
        if any(value is not None for value in slot_labels):
            normalized = [value if value is not None else [] for value in slot_labels]
            batch["slot_labels"] = self._pad_2d(
                normalized,
                self.slot_count,
                self.label_pad_token_id,
            )

        for key in ("first_subword_indices", "last_subword_indices"):
            values = [feature.get(key) for feature in features]
            if any(value is not None for value in values):
                normalized = [value if value is not None else [] for value in values]
                batch[key] = self._pad_1d(normalized, 0)

        return batch

    def _pad_1d(self, values: list[list[int]], pad_value: int) -> torch.Tensor:
        max_length = max((len(value) for value in values), default=0)
        padded = [
            list(value) + [pad_value] * (max_length - len(value)) for value in values
        ]
        return torch.tensor(padded, dtype=torch.long)

    def _pad_2d(
        self,
        values: list[list[list[int]]],
        slot_count: int,
        pad_value: int,
    ) -> torch.Tensor:
        max_length = max((len(value) for value in values), default=0)
        padded: list[list[list[int]]] = []
        pad_row = [pad_value] * slot_count
        for value in values:
            rows = [list(row) for row in value]
            rows.extend([pad_row] * (max_length - len(rows)))
            padded.append(rows)
        return torch.tensor(padded, dtype=torch.long)


class PooledTokenClassifier(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        head: str,
        pooling: str,
        labels: list[str] | None = None,
        slots: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pooling = pooling
        self.labels = list(labels or [])
        self.slots = dict(slots or {})
        self.slot_names = list(self.slots)
        self.hidden_size = hidden_size_from_config(encoder.config)
        classifier_input = self.hidden_size * (2 if pooling == "first_last" else 1)

        if head == "combined":
            if not self.labels:
                raise ValueError("combined PooledTokenClassifier requires labels")
            self.classifier = nn.Linear(classifier_input, len(self.labels))
            self.classifiers = None
        elif head == "factored":
            if not self.slots:
                raise ValueError("factored PooledTokenClassifier requires slots")
            self.classifier = None
            self.classifiers = nn.ModuleDict(
                {
                    slot_output_name(slot): nn.Linear(classifier_input, len(values))
                    for slot, values in self.slots.items()
                }
            )
        else:
            raise ValueError(f"unknown head: {head}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        first_subword_indices: torch.Tensor | None = None,
        last_subword_indices: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        slot_labels: torch.Tensor | None = None,
        **_: object,
    ) -> TokenClassifierOutput:
        outputs = self.encoder(
            **encoder_kwargs(self.encoder, input_ids, attention_mask, token_type_ids),
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        representations = self._pooled_representations(
            hidden,
            first_subword_indices,
            last_subword_indices,
        )
        logits = self._head_logits(representations)
        loss = self._loss(logits, labels, slot_labels)
        return TokenClassifierOutput(loss=loss, logits=logits)

    def _pooled_representations(
        self,
        hidden: torch.Tensor,
        first_subword_indices: torch.Tensor | None,
        last_subword_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pooling != "first_last":
            return hidden
        if first_subword_indices is None or last_subword_indices is None:
            raise ValueError("first_last pooling requires subword index tensors")
        first = self._gather_hidden(hidden, first_subword_indices)
        last = self._gather_hidden(hidden, last_subword_indices)
        return torch.cat([first, last], dim=-1)

    def _gather_hidden(self, hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        safe_indices = indices.clamp(min=0, max=hidden.shape[1] - 1)
        expanded = safe_indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return torch.gather(hidden, 1, expanded)

    def _head_logits(self, representations: torch.Tensor) -> torch.Tensor | tuple:
        if self.head == "combined":
            return self.classifier(representations)  # type: ignore[operator]
        assert self.classifiers is not None
        return tuple(
            self.classifiers[slot_output_name(slot)](representations)
            for slot in self.slot_names
        )

    def _loss(
        self,
        logits: torch.Tensor | tuple,
        labels: torch.Tensor | None,
        slot_labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.head == "combined":
            if labels is None:
                return None
            assert isinstance(logits, torch.Tensor)
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=LABEL_PAD_ID,
            )

        if slot_labels is None:
            return None
        losses: list[torch.Tensor] = []
        assert isinstance(logits, tuple)
        for slot_index, slot_logits in enumerate(logits):
            slot_gold = slot_labels[:, :, slot_index]
            if not torch.any(slot_gold != LABEL_PAD_ID):
                continue
            losses.append(
                F.cross_entropy(
                    slot_logits.reshape(-1, slot_logits.shape[-1]),
                    slot_gold.reshape(-1),
                    ignore_index=LABEL_PAD_ID,
                )
            )
        if not losses:
            first_logits = logits[0]
            return first_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def full_sequence_logits(self, hidden: torch.Tensor) -> torch.Tensor | tuple:
        if self.pooling != "first_last":
            return self._head_logits(hidden)

        if self.head == "combined":
            assert self.classifier is not None
            return self._first_last_parts(hidden, self.classifier)

        assert self.classifiers is not None
        return tuple(
            self._first_last_parts(hidden, self.classifiers[slot_output_name(slot)])
            for slot in self.slot_names
        )

    def _first_last_parts(self, hidden: torch.Tensor, classifier: nn.Linear) -> torch.Tensor:
        left_weight = classifier.weight[:, : self.hidden_size]
        right_weight = classifier.weight[:, self.hidden_size :]
        first_part = torch.matmul(hidden, left_weight.transpose(0, 1))
        last_part = torch.matmul(hidden, right_weight.transpose(0, 1)) + classifier.bias
        return torch.stack([first_part, last_part], dim=2)


class FullSequenceExportWrapper(nn.Module):
    def __init__(self, model: PooledTokenClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        outputs = self.model.encoder(
            **encoder_kwargs(
                self.model.encoder,
                input_ids,
                attention_mask,
                token_type_ids,
            ),
            return_dict=True,
        )
        return self.model.full_sequence_logits(outputs.last_hidden_state)


def create_custom_model(
    *,
    model_name: str,
    head: str,
    pooling: str,
    labels: list[str] | None = None,
    slots: dict[str, list[str]] | None = None,
) -> PooledTokenClassifier:
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(model_name)
    return PooledTokenClassifier(
        encoder=encoder,
        head=head,
        pooling=pooling,
        labels=labels,
        slots=slots,
    )


def load_custom_model(model_dir: Path, head_config: dict) -> PooledTokenClassifier:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_dir)
    encoder = AutoModel.from_config(config)
    model = PooledTokenClassifier(
        encoder=encoder,
        head=head_config["head"],
        pooling=head_config["pooling"],
        labels=head_config.get("labels"),
        slots=head_config.get("slots"),
    )
    state_path = model_dir / "pytorch_model.bin"
    if not state_path.exists():
        raise FileNotFoundError(f"missing custom model weights: {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def save_custom_model(model: PooledTokenClassifier, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
