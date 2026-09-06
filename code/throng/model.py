from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from throng import NUM_CLASSES
from throng.types import GroupRecord

from .relations import RelationExtractor, WindowRelations, diagnostics, merge_windows


@dataclass(frozen=True)
class VariantSpec:
    relations: tuple[bool, bool, bool] = (True, True, True)
    use_hypergraph: bool = True
    temporal: str = "gtda"
    dynamic_relations: bool = True


GTDA_VARIANT = "gtda_temporal"
MODEL_VARIANTS: dict[str, VariantSpec] = {
    GTDA_VARIANT: VariantSpec(),
}


def _hypergraph_messages(
    x: torch.Tensor,
    propagation: torch.Tensor,
) -> torch.Tensor:
    if propagation.device != x.device or propagation.dtype != x.dtype:
        propagation = propagation.to(device=x.device, dtype=x.dtype)
    return propagation @ x


def _batched_hypergraph_messages(
    x: torch.Tensor,
    propagation: torch.Tensor,
) -> torch.Tensor:
                                                         

                                                                        
                                                                            
                                                                        
       

    if propagation.device != x.device or propagation.dtype != x.dtype:
        propagation = propagation.to(device=x.device, dtype=x.dtype)
    return torch.matmul(propagation, x.unsqueeze(2))


def _batched_relation_projection(
    messages: torch.Tensor,
    projections: Sequence[nn.Module],
) -> torch.Tensor:
                                                                               

                                                                           
                                                                           
                                                                            
                                       
       

    weights = torch.stack(
        [projection.weight for projection in projections], dim=0
    ).to(dtype=messages.dtype)
    return torch.einsum("bwrui,roi->bwruo", messages, weights)


def _autocast_cache_dtype(device: torch.device) -> str:
                                                                           

    if device.type != "cuda":
        return str(torch.float32)
    try:
        enabled = bool(torch.is_autocast_enabled())
    except TypeError:                                                      
        enabled = bool(torch.is_autocast_enabled(device.type))
    if not enabled:
        return str(torch.float32)
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is not None:
        try:
            return str(getter(device.type))
        except (TypeError, RuntimeError):
            pass
    fallback_getter = getattr(torch, "get_autocast_gpu_dtype", None)
    if fallback_getter is not None:
        return str(fallback_getter())
    return str(torch.float16)


def _logit_tensor(value: float | Sequence[float], size: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if size is not None:
        if tensor.ndim == 0:
            tensor = tensor.repeat(size)
        if tensor.shape != (size,):
            raise ValueError(f"Expected {size} residual scales, found {tuple(tensor.shape)}")
    return torch.logit(tensor.clamp(1e-4, 1.0 - 1e-4))


class RelationHypergraphLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        dropout: float,
        relation_scale_init: float | Sequence[float] = 0.10,
        relation_scale_max: float | Sequence[float] = 1.0,
    ):
        super().__init__()
        self.relation_projections = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(3)]
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.relation_scale_logit = nn.Parameter(
            _logit_tensor(relation_scale_init, size=3)
        )
        maximum = torch.as_tensor(relation_scale_max, dtype=torch.float32)
        if maximum.ndim == 0:
            maximum = maximum.repeat(3)
        if (
            maximum.shape != (3,)
            or torch.any(maximum <= 0.0)
            or torch.any(maximum > 1.0)
        ):
            raise ValueError("relation_scale_max must contain three values in (0, 1]")
        self.register_buffer("relation_scale_max", maximum)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        relations: WindowRelations,
        enabled: tuple[bool, bool, bool],
    ) -> torch.Tensor:
        enabled_tensor = torch.tensor(enabled, dtype=x.dtype, device=x.device)
        evidence = relations.evidence.to(x.device) * enabled_tensor
        available = tuple(
            enabled[index] and bool(relations.by_type[index]) for index in range(3)
        )
        availability = torch.tensor(
            available,
            device=x.device,
            dtype=torch.bool,
        )
        if not availability.any():
            return x
        logits = self.gate(torch.cat((x.mean(0), evidence), dim=0))
        logits = logits.masked_fill(~availability, -1e4)
        gates = logits.softmax(0)
        relation_message = x.new_zeros(x.shape)
        for relation_type in range(3):
            if not available[relation_type]:
                continue
            message = _hypergraph_messages(x, relations.propagation[relation_type])
            relation_message = relation_message + (
                gates[relation_type]
                * self.relation_scale_max[relation_type]
                * torch.sigmoid(self.relation_scale_logit[relation_type])
                * self.relation_projections[relation_type](message)
            )
        return x + self.dropout(F.silu(relation_message))

    def forward_windows(
        self,
        x: torch.Tensor,
        evidence: torch.Tensor,
        propagation: torch.Tensor,
        available: torch.Tensor,
        enabled: tuple[bool, bool, bool],
    ) -> torch.Tensor:
                                                                                     

        enabled_tensor = torch.tensor(enabled, dtype=x.dtype, device=x.device)
        evidence = evidence * enabled_tensor
        availability = available & torch.tensor(enabled, dtype=torch.bool, device=x.device)
        logits = self.gate(torch.cat((x.mean(1), evidence), dim=-1))
        logits = logits.masked_fill(~availability, -1e4)
        gates = logits.softmax(-1)
        gates = gates * availability.any(-1, keepdim=True).to(gates.dtype)
        relation_message = x.new_zeros(x.shape)
        for relation_type in range(3):
            message = torch.matmul(propagation[:, relation_type], x)
            relation_message = relation_message + (
                gates[:, relation_type, None, None]
                * self.relation_scale_max[relation_type]
                * torch.sigmoid(self.relation_scale_logit[relation_type])
                * self.relation_projections[relation_type](message)
            )
        return x + self.dropout(F.silu(relation_message))

    def forward_batch(
        self,
        x: torch.Tensor,
        evidence: torch.Tensor,
        propagation: torch.Tensor,
        available: torch.Tensor,
        valid_users: torch.Tensor,
        enabled: tuple[bool, bool, bool],
    ) -> torch.Tensor:
                                                                                 

        enabled_tensor = torch.tensor(enabled, dtype=x.dtype, device=x.device)
        availability = available & torch.tensor(
            enabled, dtype=torch.bool, device=x.device
        )
        evidence = evidence * enabled_tensor
        valid = valid_users[:, None, :, None].to(x.dtype)
        summary = (x * valid).sum(2) / valid.sum(2).clamp_min(1.0)
        logits = self.gate(torch.cat((summary, evidence), dim=-1))
        logits = logits.masked_fill(~availability, -1e4)
        gates = logits.softmax(-1)
        gates = gates * availability.any(-1, keepdim=True).to(gates.dtype)
        scales = self.relation_scale_max * torch.sigmoid(
            self.relation_scale_logit
        )
        relation_indices = tuple(index for index, value in enumerate(enabled) if value)
        if not relation_indices:
            return x * valid
        if relation_indices == (0, 1, 2):
            selected_propagation = propagation
            selected_gates = gates[:, :, :3]
            selected_scales = scales
            selected_projections = self.relation_projections
        else:
            index = torch.tensor(
                relation_indices, dtype=torch.long, device=x.device
            )
            selected_propagation = propagation.index_select(2, index)
            selected_gates = gates.index_select(-1, index)
            selected_scales = scales.index_select(0, index)
            selected_projections = [
                self.relation_projections[relation_type]
                for relation_type in relation_indices
            ]
        messages = _batched_hypergraph_messages(x, selected_propagation)
        projected = _batched_relation_projection(messages, selected_projections)
        relation_message = (
            projected
            * selected_gates.unsqueeze(-1).unsqueeze(-1)
            * selected_scales.view(1, 1, -1, 1, 1)
        ).sum(2)
        output = x + self.dropout(F.silu(relation_message))
        return output * valid


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        hidden: int,
        dropout: float,
        mode: str,
        bins: int = 8,
        residual_scale_init: float = 0.10,
        temporal_mix_init: float = 0.25,
        residual_scale_max: float = 1.0,
        temporal_mix_max: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.temporal_mix_logit = nn.Parameter(_logit_tensor(temporal_mix_init))
        self.raw_norm = nn.LayerNorm(hidden)
        if not 0.0 < temporal_mix_max <= 1.0:
            raise ValueError("temporal_mix_max must lie in (0, 1]")
        self.register_buffer(
            "temporal_mix_max", torch.tensor(float(temporal_mix_max))
        )
        if mode != "gtda":
            raise ValueError(f"Unknown temporal encoder: {mode}")
        self.encoder = nn.Identity()
        self.gtda_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.gtda_gate = nn.Linear(hidden * 2, hidden)
        self.gtda_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        raw_mean = windows.mean(1)
        if windows.shape[1] > 1:
            length = windows.shape[1]
            mean_difference = (windows[:, -1] - windows[:, 0]) / float(length - 1)
            last_difference = windows[:, -1] - windows[:, -2]
        else:
            mean_difference = windows.new_zeros(windows.shape[0], windows.shape[2])
            last_difference = mean_difference
        base = self.raw_norm(raw_mean)
        change = self.gtda_projection(
            torch.cat((mean_difference, last_difference), dim=-1)
        )
        gate = torch.sigmoid(self.gtda_gate(torch.cat((base, change), dim=-1)))
        temporal_mix = self.temporal_mix_max * torch.sigmoid(self.temporal_mix_logit)
        pooled = base + temporal_mix * gate * self.gtda_dropout(change)
        return self.norm(pooled)


class THRONG(nn.Module):
    def __init__(
        self,
        hidden: int = 64,
        dropout: float = 0.1,
        bins: int = 8,
        hypergraph_layers: int = 2,
        relation_dropout: float = 0.1,
        template_cosine_threshold: float = 0.75,
        shift_cosine_threshold: float = 0.70,
        shift_minimum_norm: float = 0.10,
        relation_scale_init: float | Sequence[float] = 0.10,
        relation_context_scale_init: float | Sequence[float] = 0.10,
        temporal_residual_scale_init: float = 0.10,
        temporal_mix_init: float = 0.25,
        relation_scale_max: float | Sequence[float] = 1.0,
        relation_context_scale_max: float | Sequence[float] = 1.0,
        temporal_residual_scale_max: float = 1.0,
        temporal_mix_max: float = 1.0,
        variant: str = GTDA_VARIANT,
    ):
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise KeyError(f"Unknown THRONG variant: {variant}")
        self.hidden = hidden
        self.bins = bins
        self.variant = variant
        self.relation_dropout = float(relation_dropout)
        self.spec = MODEL_VARIANTS[variant]
        self.extractor = RelationExtractor(
            bins=bins,
            cosine_threshold=shift_cosine_threshold,
            minimum_norm=shift_minimum_norm,
            template_cosine_threshold=template_cosine_threshold,
        )
        self.user_projection = nn.Linear(49, hidden)
        self.session_projection = nn.Linear(15, hidden)
        self.evidence_projection = nn.Linear(4, hidden)
        self.relation_context_scale_logit = nn.Parameter(
            _logit_tensor(relation_context_scale_init, size=3)
        )
        context_maximum = torch.as_tensor(
            relation_context_scale_max, dtype=torch.float32
        )
        if context_maximum.ndim == 0:
            context_maximum = context_maximum.repeat(3)
        if (
            context_maximum.shape != (3,)
            or torch.any(context_maximum <= 0.0)
            or torch.any(context_maximum > 1.0)
        ):
            raise ValueError(
                "relation_context_scale_max must contain three values in (0, 1]"
            )
        self.register_buffer("relation_context_scale_max", context_maximum)
        self.hypergraph_layers = nn.ModuleList(
            [
                RelationHypergraphLayer(
                    hidden,
                    dropout,
                    relation_scale_init=relation_scale_init,
                    relation_scale_max=relation_scale_max,
                )
                for _ in range(hypergraph_layers)
            ]
        )
        self.window_fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = TemporalEncoder(
            hidden,
            dropout,
            self.spec.temporal,
            bins=bins,
            residual_scale_init=temporal_residual_scale_init,
            temporal_mix_init=temporal_mix_init,
            residual_scale_max=temporal_residual_scale_max,
            temporal_mix_max=temporal_mix_max,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )
        self._feature_cache: dict[
            tuple[int, int], tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]
        ] = {}
        self._window_embedding_cache: dict[
            tuple[tuple[int, ...], int, str, int | None, str], torch.Tensor
        ] = {}
        self._window_embedding_cache_enabled = True

    def _window_features(
        self,
        record: GroupRecord,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        cache_key = (record.group_index, self.bins)
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            return cached
        number_of_users = len(record.users)
        sessions = record.sessions.float()
        session_bins = record.session_bins.long()
        session_users = record.session_users.long()
        valid = (
            (session_bins >= 0)
            & (session_bins < self.bins)
            & (session_users >= 0)
            & (session_users < number_of_users)
        )
        valid_sessions = sessions[valid]
        valid_bins = session_bins[valid]
        valid_users = session_users[valid]

        flattened = valid_bins * number_of_users + valid_users
        sums = torch.zeros(
            (self.bins * number_of_users, 15), dtype=torch.float32
        )
        counts = torch.zeros(self.bins * number_of_users, dtype=torch.float32)
        if len(valid_users):
            sums.index_add_(0, flattened, valid_sessions)
            counts.index_add_(
                0, flattened, torch.ones(len(valid_users), dtype=torch.float32)
            )
        means = (
            sums / counts.clamp_min(1.0).unsqueeze(-1)
        ).view(self.bins, number_of_users, 15)
        window_counts = counts.view(self.bins, number_of_users)
        activity = window_counts / window_counts.sum(1, keepdim=True).clamp_min(1.0)
        user_features = record.users.float().unsqueeze(0).expand(
            self.bins, -1, -1
        )
        node_features = torch.cat((user_features, means, activity.unsqueeze(-1)), dim=-1)

        session_sums = torch.zeros((self.bins, 15), dtype=torch.float32)
        session_counts = torch.zeros(self.bins, dtype=torch.float32)
        if len(valid_bins):
            session_sums.index_add_(0, valid_bins, valid_sessions)
            session_counts.index_add_(
                0, valid_bins, torch.ones(len(valid_bins), dtype=torch.float32)
            )
        session_means = session_sums / session_counts.clamp_min(1.0).unsqueeze(-1)
        result = (
            tuple(node_features[window] for window in range(self.bins)),
            tuple(session_means[window] for window in range(self.bins)),
        )
        self._feature_cache[cache_key] = result
        return result

    def _temporal_record(self, record: GroupRecord) -> GroupRecord:
                                                                       

                                                                             
                                                                              
                                                                           
           

        if self.bins == 8:
            return record
        bins = torch.floor(record.session_times.float() * self.bins).long()
        bins = torch.minimum(bins, torch.full_like(bins, self.bins - 1))
        return replace(record, session_bins=bins)

    def _relations(self, record: GroupRecord) -> tuple[WindowRelations, ...]:
        windows = self.extractor.extract(record)
        if self.spec.dynamic_relations:
            return windows
        merged = merge_windows(windows, len(record.users))
        return tuple(
            replace(merged, active_users=window.active_users)
            for window in windows
        )

    def _relation_tensors(
        self,
        record: GroupRecord,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                                                              

                                                                         
                                                                          
                                     
           
        relation_windows = self._relations(record)
        evidence = torch.stack(
            [relations.evidence for relations in relation_windows]
        )
        propagation = torch.stack(
            [torch.stack(relations.propagation) for relations in relation_windows]
        )
        available = torch.tensor(
            [
                [bool(relations.by_type[index]) for index in range(3)]
                for relations in relation_windows
            ],
            dtype=torch.bool,
        )
        active_mask = torch.zeros(
            (self.bins, len(record.users)),
            dtype=torch.bool,
        )
        for window, relations in enumerate(relation_windows):
            active_mask[window, relations.active_users] = True
        return evidence, propagation, available, active_mask

    def _batch_inputs(
        self,
        records: Sequence[GroupRecord],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
                                                                      

        maximum_users = max(len(record.users) for record in records)
        batch_size = len(records)

        pin_memory = device.type == "cuda"

        def staging_zeros(shape, dtype):
            if not pin_memory:
                return torch.zeros(shape, dtype=dtype)
            try:
                return torch.zeros(shape, dtype=dtype, pin_memory=True)
            except RuntimeError:
                return torch.zeros(shape, dtype=dtype)

        node_inputs = staging_zeros(
            (
                batch_size,
                self.bins,
                maximum_users,
                self.user_projection.in_features,
            ),
            dtype=torch.float32,
        )
        session_inputs = staging_zeros(
            (batch_size, self.bins, self.session_projection.in_features),
            dtype=torch.float32,
        )
        evidence = staging_zeros((batch_size, self.bins, 3), dtype=torch.float32)
        propagation = staging_zeros(
            (batch_size, self.bins, 3, maximum_users, maximum_users),
            dtype=torch.float32,
        )
        available = staging_zeros((batch_size, self.bins, 3), dtype=torch.bool)
        active_mask = staging_zeros(
            (batch_size, self.bins, maximum_users), dtype=torch.bool
        )
        valid_users = staging_zeros((batch_size, maximum_users), dtype=torch.bool)

        for index, record in enumerate(records):
            node_features, session_means = self._window_features(record)
            nodes = torch.stack(node_features)
            sessions = torch.stack(session_means)
            relation_evidence, relation_propagation, relation_available, relation_active = (
                self._relation_tensors(record)
            )
            user_count = len(record.users)
            node_inputs[index, :, :user_count] = nodes
            session_inputs[index] = sessions
            evidence[index] = relation_evidence
            propagation[index, :, :, :user_count, :user_count] = relation_propagation
            available[index] = relation_available
            active_mask[index, :, :user_count] = relation_active
            valid_users[index, :user_count] = True

        result = (
            node_inputs,
            session_inputs,
            evidence,
            propagation,
            available,
            active_mask,
            valid_users,
        )
        if device.type == "cpu":
            return result
        return tuple(
            value.to(device=device, non_blocking=True) for value in result
        )

    def window_embeddings_batch(
        self,
        records: Sequence[GroupRecord],
    ) -> torch.Tensor:
        if not records:
            raise ValueError("THRONG requires at least one group per batch")
        records = [self._temporal_record(record) for record in records]
        device = self.user_projection.weight.device
        cache_key = None
        if (
            self._window_embedding_cache_enabled
            and not self.training
            and not torch.is_grad_enabled()
        ):
            cache_dtype = _autocast_cache_dtype(device)
            cache_key = (
                tuple(int(record.group_index) for record in records),
                self.bins,
                device.type,
                device.index,
                cache_dtype,
            )
            cached = self._window_embedding_cache.get(cache_key)
            if cached is not None:
                return cached
        (
            node_inputs,
            session_inputs,
            evidence,
            propagation,
            available,
            active_mask,
            valid_users,
        ) = self._batch_inputs(records, device)

        if self.training and self.relation_dropout > 0.0:
            keep = (
                torch.rand(evidence.shape, device=device) >= self.relation_dropout
            )
            evidence = evidence * keep.to(evidence.dtype)
            propagation = propagation * keep[..., None, None].to(
                propagation.dtype
            )
            available = available & keep
        valid = valid_users[:, None, :, None].to(node_inputs.dtype)
        nodes = self.user_projection(node_inputs) * valid
        enabled_mask = torch.tensor(self.spec.relations, dtype=torch.float32, device=device)
        if self.spec.use_hypergraph:
            for layer in self.hypergraph_layers:
                nodes = layer.forward_batch(
                    nodes,
                    evidence,
                    propagation,
                    available,
                    valid_users,
                    self.spec.relations,
                )
        active_count = active_mask.sum(2).clamp_min(1).to(nodes.dtype)
        pooled_users = (
            nodes * active_mask.unsqueeze(-1)
        ).sum(2) / active_count.unsqueeze(-1)
        valid_count = valid_users.sum(1).clamp_min(1).to(nodes.dtype)
        fallback = (nodes * valid).sum(2) / valid_count[:, None, None]
        pooled_users = torch.where(
            active_mask.any(2).unsqueeze(-1), pooled_users, fallback
        )
        pooled_sessions = self.session_projection(session_inputs)
        active_fraction = active_count / valid_count[:, None]
        relation_evidence = (
            evidence
            * enabled_mask
            * self.relation_context_scale_max
            * torch.sigmoid(self.relation_context_scale_logit)
        )
        if not self.spec.use_hypergraph:
            relation_evidence = torch.zeros_like(relation_evidence)
        relation_context = self.evidence_projection(
            torch.cat((relation_evidence, active_fraction.unsqueeze(-1)), dim=-1)
        )
        result = self.window_fusion(
            torch.cat((pooled_users, pooled_sessions, relation_context), dim=-1)
        )
        if cache_key is not None:
            result = result.detach()
            self._window_embedding_cache[cache_key] = result
        return result

    def window_embeddings(self, record: GroupRecord) -> torch.Tensor:
        return self.window_embeddings_batch([record])[0]

    def encode_batch(self, records: Sequence[GroupRecord]) -> torch.Tensor:
        windows = self.window_embeddings_batch(records)
        return self.temporal_encoder(windows)

    def forward_batch(self, records: Sequence[GroupRecord]) -> torch.Tensor:
        return self.head(self.encode_batch(records))

    def _window_embedding_cache_key(
        self, records: Sequence[GroupRecord]
    ) -> tuple[tuple[int, ...], int, str, int | None, str]:
        device = self.user_projection.weight.device
        records = [self._temporal_record(record) for record in records]
        return (
            tuple(int(record.group_index) for record in records),
            self.bins,
            device.type,
            device.index,
            _autocast_cache_dtype(device),
        )

    def cached_window_embeddings_batch(
        self, records: Sequence[GroupRecord]
    ) -> torch.Tensor:
                                                                             

        if not records:
            raise ValueError("THRONG requires at least one group per batch")
        key = self._window_embedding_cache_key(records)
        cached = self._window_embedding_cache.get(key)
        if cached is None:
            raise RuntimeError(
                "Requested model-only inference before fused windows were cached"
            )
        return cached

    def forward_cached_batch(self, records: Sequence[GroupRecord]) -> torch.Tensor:
                                                                           

        return self.head(
            self.temporal_encoder(self.cached_window_embeddings_batch(records))
        )

    def forward_record(self, record: GroupRecord) -> torch.Tensor:
        return self.forward_batch([record])[0]

    def train(self, mode: bool = True):
        if mode and hasattr(self, "_window_embedding_cache"):
            self._window_embedding_cache.clear()
        return super().train(mode)

    def load_state_dict(self, *args, **kwargs):
        self.clear_runtime_caches()
        return super().load_state_dict(*args, **kwargs)

    def _apply(self, fn):
        if hasattr(self, "_window_embedding_cache"):
            self.clear_runtime_caches()
        return super()._apply(fn)

    def clear_runtime_caches(self) -> None:
                                                                                     

        self._feature_cache.clear()
        self._window_embedding_cache.clear()
        self.extractor._cache.clear()

    @contextmanager
    def cache_window_embeddings(self, enabled: bool) -> Iterator[None]:
                                                                         

        previous = self._window_embedding_cache_enabled
        self._window_embedding_cache_enabled = bool(enabled)
        if not enabled:
            self._window_embedding_cache.clear()
        try:
            yield
        finally:
            self._window_embedding_cache_enabled = previous
            if not previous:
                self._window_embedding_cache.clear()

    def loss(self, logits: torch.Tensor, labels: torch.Tensor, records=None) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def relation_diagnostics(self, record: GroupRecord) -> dict[str, float]:
        return diagnostics(self.extractor.extract(record))
