from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from throng import NUM_CLASSES
from throng.types import GroupRecord

from .model import THRONG


@dataclass(frozen=True)
class AdaptiveOutput:
    predictions: torch.Tensor
    probabilities: torch.Tensor
    crowd_scores: torch.Tensor


def freeze_encoder(model: THRONG) -> None:
    modules = (
        model.user_projection,
        model.session_projection,
        model.evidence_projection,
        model.hypergraph_layers,
        model.window_fusion,
        model.temporal_encoder,
    )
    for module in modules:
        module.requires_grad_(False)
        module.eval()


def fit_few_shot_head(
    model: THRONG,
    support_records: Sequence[GroupRecord],
    support_labels: torch.Tensor,
    steps: int = 100,
    learning_rate: float = 1e-2,
) -> nn.Linear:
                                                                      

    freeze_encoder(model)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        embeddings = model.encode_batch(support_records).detach()
    labels = support_labels.to(device)
    head = nn.Linear(model.hidden, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(embeddings), labels)
        loss.backward()
        optimizer.step()
    return head


@torch.no_grad()
def few_shot_predict(
    model: THRONG,
    head: nn.Linear,
    records: Sequence[GroupRecord],
) -> AdaptiveOutput:
    embeddings = model.encode_batch(records)
    probabilities = head(embeddings).softmax(-1)
    crowd_scores = probabilities[:, 1]
    return AdaptiveOutput(probabilities.argmax(-1), probabilities, crowd_scores)


@torch.no_grad()
def zero_shot_filter(
    model: THRONG,
    records: Sequence[GroupRecord],
    crowd_threshold: float | None = None,
) -> AdaptiveOutput:
                                                                                  

                                                                               
                                              
       

    if crowd_threshold is None:
        raise ValueError("zero_shot_filter requires a source-calibrated threshold")
    model.eval()
    logits = model.forward_batch(records)
    probabilities = logits.softmax(-1)
    crowd_scores = []
    for record in records:
        windows = model.extractor.extract(record)
        evidence = torch.stack([window.evidence for window in windows])
        score = evidence[:, :2].amax(1).mean()
        crowd_scores.append(score)
    crowd_scores_tensor = torch.stack(crowd_scores).to(probabilities.device)
    predictions = torch.where(
        crowd_scores_tensor >= crowd_threshold,
        torch.ones_like(crowd_scores_tensor, dtype=torch.long),
        torch.zeros_like(crowd_scores_tensor, dtype=torch.long),
    )
    return AdaptiveOutput(predictions, probabilities, crowd_scores_tensor)


@torch.no_grad()
def calibrate_zero_shot_threshold(
    model: THRONG,
    records: Sequence[GroupRecord],
    labels: torch.Tensor,
) -> float:
                                                                                

    if not records:
        raise ValueError("Threshold calibration requires source records")
    device = next(model.parameters()).device
    labels = labels.to(device).long().reshape(-1)
    if labels.numel() != len(records) or labels.unique().numel() < 2:
        raise ValueError("Threshold calibration requires both source classes")
    model.eval()
    scores = []
    for record in records:
        evidence = torch.stack(
            [window.evidence for window in model.extractor.extract(record)]
        )
        scores.append(evidence[:, :2].amax(1).mean())
    values = torch.stack(scores)
    candidates = torch.unique(torch.cat((values, values.new_tensor([0.0, 1.0]))))
    best = None
    for threshold in candidates:
        predictions = (values >= threshold).long()
        f1_values = []
        for class_id in (0, 1):
            tp = ((labels == class_id) & (predictions == class_id)).sum().float()
            fp = ((labels != class_id) & (predictions == class_id)).sum().float()
            fn = ((labels == class_id) & (predictions != class_id)).sum().float()
            denominator = 2.0 * tp + fp + fn
            f1_values.append(
                2.0 * tp / denominator
                if denominator > 0
                else values.new_tensor(0.0)
            )
        score = torch.stack(f1_values).mean().item()
        key = (score, -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return best[1]
