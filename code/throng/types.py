from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroupRecord:
    group_index: int
    label: int
    sessions: torch.Tensor
    session_times: torch.Tensor
    session_bins: torch.Tensor
    session_users: torch.Tensor
    session_targets: torch.Tensor
    session_templates: torch.Tensor
    users: torch.Tensor
    targets: torch.Tensor
    relation_edges: torch.Tensor
    relation_types: torch.Tensor
    relation_weights: torch.Tensor
    user_target_edges: torch.Tensor
    user_target_weights: torch.Tensor
