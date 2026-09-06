from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F

from throng.types import GroupRecord


DESTINATION = 0
TEMPLATE = 1
SHIFT = 2
RELATION_NAMES = ("destination_recurrence", "template_reuse", "shift_alignment")


@dataclass(frozen=True)
class Hyperedge:
    members: torch.Tensor
    weight: float


@dataclass(frozen=True)
class WindowRelations:
    by_type: tuple[tuple[Hyperedge, ...], tuple[Hyperedge, ...], tuple[Hyperedge, ...]]
    evidence: torch.Tensor
    active_users: torch.Tensor
    propagation: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _members(users: torch.Tensor, mask: torch.Tensor) -> set[int]:
    return set(int(value) for value in torch.unique(users[mask]).tolist())


def _edge(members: Iterable[int], weight: float) -> Hyperedge:
    return Hyperedge(
        members=torch.tensor(sorted(set(members)), dtype=torch.long),
        weight=float(max(0.0, min(1.0, weight))),
    )


def _destination_edges(record: GroupRecord, window: int, active_count: int) -> list[Hyperedge]:
    current = record.session_bins == window
    past = record.session_bins < window
    if window == 0 or not current.any() or not past.any():
        return []
    edges = []
    for target in torch.unique(record.session_targets[current]).tolist():
        current_members = _members(
            record.session_users,
            current & (record.session_targets == int(target)),
        )
        if len(current_members) < 2:
            continue
        past_members = _members(
            record.session_users,
            past & (record.session_targets == int(target)),
        )
        recurrence = _jaccard(current_members, past_members)
        if recurrence <= 0.0:
            continue
        support = len(current_members) / max(active_count, 1)
        edges.append(_edge(current_members, (recurrence * support) ** 0.5))
    return edges


def _template_edges(
    record: GroupRecord,
    window: int,
    active_count: int,
    cosine_threshold: float,
) -> list[Hyperedge]:
    current = record.session_bins == window
    if not current.any():
        return []
    users = sorted(_members(record.session_users, current))
    representations = {
        user: record.sessions[current & (record.session_users == user)]
        .float()
        .mean(0)
        for user in users
    }
    accepted = []
    for offset, left in enumerate(users):
        for right in users[offset + 1 :]:
            similarity = float(
                F.cosine_similarity(
                    representations[left], representations[right], dim=0
                )
            )
            if similarity >= cosine_threshold:
                accepted.append((left, right, similarity))
    edges = []
    for component in _components(users, accepted):
        similarities = [
            similarity
            for left, right, similarity in accepted
            if left in component and right in component
        ]
        support = len(component) / max(active_count, 1)
        edges.append(
            _edge(component, support * sum(similarities) / len(similarities))
        )
    return edges


def _user_shift(record: GroupRecord, user: int, window: int) -> torch.Tensor | None:
    current = (record.session_users == user) & (record.session_bins == window)
    history = (record.session_users == user) & (record.session_bins < window)
    if not current.any() or not history.any():
        return None
    return record.sessions[current].float().mean(0) - record.sessions[history].float().mean(0)


def _components(nodes: list[int], accepted_pairs: list[tuple[int, int, float]]):
    parent = {node: node for node in nodes}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right, _ in accepted_pairs:
        union(left, right)
    groups: dict[int, list[int]] = {}
    for node in nodes:
        groups.setdefault(find(node), []).append(node)
    return [members for members in groups.values() if len(members) >= 2]


def _shift_edges(
    record: GroupRecord,
    window: int,
    active_users: list[int],
    cosine_threshold: float,
    minimum_norm: float,
) -> list[Hyperedge]:
    if window == 0:
        return []
    shifts = {}
    for user in active_users:
        shift = _user_shift(record, user, window)
        if shift is not None and float(torch.linalg.vector_norm(shift)) >= minimum_norm:
            shifts[user] = shift
    users = sorted(shifts)
    accepted = []
    for offset, left in enumerate(users):
        for right in users[offset + 1 :]:
            similarity = float(F.cosine_similarity(shifts[left], shifts[right], dim=0))
            if similarity >= cosine_threshold:
                accepted.append((left, right, similarity))
    edges = []
    for component in _components(users, accepted):
        similarities = [
            similarity
            for left, right, similarity in accepted
            if left in component and right in component
        ]
        support = len(component) / max(len(active_users), 1)
        edges.append(_edge(component, support * sum(similarities) / len(similarities)))
    return edges


def _evidence(edges: list[Hyperedge]) -> float:
    if not edges:
        return 0.0
    complement = 1.0
    for edge in edges:
        complement *= 1.0 - edge.weight
    return 1.0 - complement


def _propagation(edges: tuple[Hyperedge, ...], number_of_users: int) -> torch.Tensor:
                                                                             

    if not edges:
        return torch.zeros((number_of_users, number_of_users), dtype=torch.float32)
    incidence = torch.zeros((number_of_users, len(edges)), dtype=torch.float32)
    for index, edge in enumerate(edges):
        incidence[edge.members, index] = float(edge.weight)
    user_degree = incidence.sum(1)
    edge_degree = incidence.sum(0)
    user_inv_sqrt = torch.where(
        user_degree > 0.0,
        user_degree.rsqrt(),
        torch.zeros_like(user_degree),
    )
    edge_inv = torch.where(
        edge_degree > 0.0,
        edge_degree.reciprocal(),
        torch.zeros_like(edge_degree),
    )
    normalized = user_inv_sqrt[:, None] * incidence * edge_inv[None, :]
    return normalized @ incidence.transpose(0, 1) * user_inv_sqrt[None, :]


class RelationExtractor:
                                                                                   

    def __init__(
        self,
        bins: int = 8,
        cosine_threshold: float = 0.70,
        minimum_norm: float = 0.10,
        template_cosine_threshold: float = 0.75,
    ):
        self.bins = bins
        self.cosine_threshold = cosine_threshold
        self.minimum_norm = minimum_norm
        self.template_cosine_threshold = template_cosine_threshold
        self._cache: dict[int, tuple[WindowRelations, ...]] = {}

    def extract(self, record: GroupRecord) -> tuple[WindowRelations, ...]:
        cached = self._cache.get(record.group_index)
        if cached is not None:
            return cached
        windows = []
        for window in range(self.bins):
            mask = record.session_bins == window
            active = torch.unique(record.session_users[mask]).long()
            active_users = [int(value) for value in active.tolist()]
            destination = _destination_edges(record, window, len(active_users))
            template = _template_edges(
                record,
                window,
                len(active_users),
                self.template_cosine_threshold,
            )
            shift = _shift_edges(
                record,
                window,
                active_users,
                self.cosine_threshold,
                self.minimum_norm,
            )
            families = (tuple(destination), tuple(template), tuple(shift))
            windows.append(
                WindowRelations(
                    by_type=families,
                    evidence=torch.tensor([_evidence(list(edges)) for edges in families], dtype=torch.float32),
                    active_users=active,
                    propagation=tuple(
                        _propagation(edges, len(record.users)) for edges in families
                    ),
                )
            )
        result = tuple(windows)
        self._cache[record.group_index] = result
        return result


def merge_windows(windows: tuple[WindowRelations, ...], number_of_users: int) -> WindowRelations:
    merged = []
    for relation_type in range(3):
        grouped: dict[tuple[int, ...], list[float]] = {}
        for window in windows:
            for edge in window.by_type[relation_type]:
                key = tuple(int(value) for value in edge.members.tolist())
                grouped.setdefault(key, []).append(edge.weight)
        merged.append(
            tuple(_edge(key, sum(weights) / len(weights)) for key, weights in grouped.items())
        )
    evidence = torch.stack([window.evidence for window in windows]).mean(0)
    return WindowRelations(
        by_type=(merged[0], merged[1], merged[2]),
        evidence=evidence,
        active_users=torch.arange(number_of_users, dtype=torch.long),
        propagation=tuple(
            _propagation(edges, number_of_users) for edges in merged
        ),
    )


def diagnostics(windows: tuple[WindowRelations, ...]) -> dict[str, float]:
    output = {}
    for relation_type, name in enumerate(RELATION_NAMES):
        output[f"{name}_edges"] = float(sum(len(window.by_type[relation_type]) for window in windows))
        output[f"{name}_evidence"] = float(
            torch.stack([window.evidence[relation_type] for window in windows]).mean()
        )
    return output
