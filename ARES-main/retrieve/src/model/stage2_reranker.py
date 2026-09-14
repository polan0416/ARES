import torch
import torch.nn as nn

from src.config.stage2_reranker import get_label_config


class Stage2TripleReranker(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError('hidden_dims must not be empty')

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_stage2):
        return self.mlp(x_stage2).reshape(-1)


def soft_bce_loss(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, soft_labels)


def compute_inference_fusion_score(
    prepared: dict,
    logits: torch.Tensor,
    config: dict,
    fusion_w1=None,
    fusion_w2=None,
) -> torch.Tensor:
    inf = config.get('inference', {})
    mode = str(inf.get('fusion_mode', 'linear')).lower()
    logits = logits.reshape(-1).float()
    stage1_probs = prepared['f_stage1'][:, 0].reshape(-1).float()

    if mode == 'logit_sum':
        stage1_logit = prepared['f_stage1'][:, 1].reshape(-1).float()
        scale = float(inf.get('stage2_logit_scale', 1.0))
        return torch.sigmoid(stage1_logit + scale * logits)

    w1 = float(fusion_w1 if fusion_w1 is not None else inf['fusion_weight_stage1'])
    w2 = float(fusion_w2 if fusion_w2 is not None else inf['fusion_weight_stage2'])
    stage2_probs = torch.sigmoid(logits)
    return w1 * stage1_probs + w2 * stage2_probs.to(dtype=torch.float32)


def num_answer_entities(sample: dict) -> int:
    a = sample.get('a_entity_in_graph') or sample.get('a_entity') or []
    n = sum(1 for x in a if str(x).strip())
    return max(1, n)


def answer_touch_targets(sample: dict, num_candidates: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cand = sample.get('candidate_triples_norm')
    if not cand or len(cand) != num_candidates:
        return torch.zeros(num_candidates, device=device, dtype=dtype)
    answers = sample.get('a_entity_in_graph') or sample.get('a_entity') or []
    ans_set = {str(x).strip() for x in answers}
    if not ans_set:
        return torch.zeros(num_candidates, device=device, dtype=dtype)
    vals = []
    for triple in cand:
        h, _, t = triple[0], triple[1], triple[2]
        h_s, t_s = str(h).strip(), str(t).strip()
        vals.append(1.0 if (h_s in ans_set or t_s in ans_set) else 0.0)
    return torch.tensor(vals, device=device, dtype=dtype)


def answer_touch_bce_loss(logits: torch.Tensor, sample: dict, config: dict) -> torch.Tensor:
    aux = config.get('answer_aux') or {}
    if not aux.get('enabled', False):
        return logits.new_tensor(0.0)

    targets = answer_touch_targets(sample, logits.numel(), logits.device, logits.dtype)
    if targets.numel() == 0:
        return logits.new_tensor(0.0)
    pos = float(targets.sum().item())
    if pos < 0.5:
        return logits.new_tensor(0.0)

    n = targets.numel()
    ratio = (n - pos) / max(pos, 1e-6)
    cap = float(aux.get('pos_weight_cap', 50.0))
    if ratio < 1e-6:
        w = 1.0
    else:
        w = min(ratio, cap)
    pos_weight = torch.tensor(w, device=logits.device, dtype=logits.dtype)
    out = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight,
    )
    boost = float(aux.get('multi_answer_touch_boost', 0.0))
    if boost > 0.0:
        n = num_answer_entities(sample)
        if n > 1:
            out = out * (1.0 + boost * float(n - 1))
    return out


def min_answer_coverage_loss(logits: torch.Tensor, sample: dict, config: dict) -> torch.Tensor:
    aux = config.get('answer_aux') or {}
    if not aux.get('enabled', False):
        return logits.new_tensor(0.0)
    if float(aux.get('min_answer_weight', 0.0)) <= 0.0:
        return logits.new_tensor(0.0)

    cand = sample.get('candidate_triples_norm')
    answers = sample.get('a_entity_in_graph') or sample.get('a_entity') or []
    if not cand or not answers:
        return logits.new_tensor(0.0)

    losses = []
    for a in answers:
        a = str(a).strip()
        idxs = []
        for i, triple in enumerate(cand):
            h, _, t = triple[0], triple[1], triple[2]
            if str(h).strip() == a or str(t).strip() == a:
                idxs.append(i)
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, device=logits.device, dtype=torch.long)
        sub = logits[idx_t]
        max_logit = sub.max()
        losses.append(
            nn.functional.binary_cross_entropy_with_logits(
                max_logit.unsqueeze(0),
                torch.ones(1, device=logits.device, dtype=logits.dtype),
            )
        )
    if not losses:
        return logits.new_tensor(0.0)
    out = torch.stack(losses).sum()
    boost = float(aux.get('multi_answer_min_cov_boost', 0.0))
    if boost > 0.0:
        n = num_answer_entities(sample)
        if n > 1:
            out = out * (1.0 + boost * float(n - 1))
    return out


def topk_listwise_surrogate_loss(logits: torch.Tensor, sample: dict, config: dict) -> torch.Tensor:
    aux = config.get('answer_aux') or {}
    if not aux.get('enabled', False):
        return logits.new_tensor(0.0)
    if float(aux.get('topk_listwise_weight', 0.0)) <= 0.0:
        return logits.new_tensor(0.0)
    if logits.numel() == 0:
        return logits.new_tensor(0.0)

    tau = float(aux.get('topk_listwise_tau', 1.0))
    tau = max(tau, 1e-3)
    k_cfg = int(aux.get('topk_listwise_target_k', 0))
    if k_cfg <= 0:
        k_cfg = int((config.get('eval') or {}).get('target_k', 200))
    k = float(min(k_cfg, int(logits.numel())))

    s = logits.reshape(-1).float()
    diff = (s.unsqueeze(0) - s.unsqueeze(1)) / tau
    soft_rank = 1.0 + torch.sigmoid(diff).sum(dim=0)
    p_in_topk = torch.sigmoid((k + 0.5 - soft_rank) / tau)

    touch_t = answer_touch_targets(sample, logits.numel(), logits.device, logits.dtype).float()
    pos = float(touch_t.sum().item())
    if pos < 0.5:
        return logits.new_tensor(0.0)

    eps = 1e-6
    n = float(touch_t.numel())
    pos_w = min((n - pos) / max(pos, eps), float(aux.get('pos_weight_cap', 50.0)))
    edge_w = torch.where(touch_t > 0.5, torch.full_like(touch_t, pos_w), torch.ones_like(touch_t))
    edge_loss = -(
        touch_t * torch.log(p_in_topk.clamp_min(eps)) +
        (1.0 - touch_t) * torch.log((1.0 - p_in_topk).clamp_min(eps))
    )
    edge_loss = (edge_w * edge_loss).mean()

    cand = sample.get('candidate_triples_norm') or []
    answers = [str(x).strip() for x in (sample.get('a_entity_in_graph') or sample.get('a_entity') or []) if str(x).strip()]
    if not cand or not answers:
        return edge_loss

    per_ans_losses = []
    for a in answers:
        idxs = []
        for i, triple in enumerate(cand):
            h, _, t = triple[0], triple[1], triple[2]
            if str(h).strip() == a or str(t).strip() == a:
                idxs.append(i)
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, device=logits.device, dtype=torch.long)
        p_hit = 1.0 - torch.prod(1.0 - p_in_topk[idx_t].clamp(0.0, 1.0))
        per_ans_losses.append(-torch.log(p_hit.clamp_min(eps)))

    if not per_ans_losses:
        return edge_loss
    ans_cov_loss = torch.stack(per_ans_losses).mean()
    fullcov_w = float(aux.get('topk_listwise_fullcov_weight', 0.5))
    return edge_loss + fullcov_w * ans_cov_loss


def _candidate_topic_distance(sample: dict, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.empty(0, device=indices.device)
    topic_dir = sample['topic_dir_distances'][indices]
    return topic_dir.min(dim=1).values


def select_pairwise_examples(sample: dict, config: dict):
    labels = sample['soft_labels']
    num_candidates = labels.numel()
    if num_candidates == 0:
        empty = torch.empty(0, dtype=torch.long, device=labels.device)
        return empty, empty

    label_cfg = get_label_config(config)
    bridge_label = float(label_cfg.get('bridge_label', label_cfg.get('silver_label', 0.7)))
    label_pos = labels == 1.0
    label_bridge = labels == bridge_label
    label_zero = labels == 0.0
    min_topic_distance = _candidate_topic_distance(
        sample,
        torch.arange(num_candidates, device=labels.device),
    )
    close_to_topic = min_topic_distance <= config['pairwise']['high_conf_min_topic_distance']

    pos_mask = label_pos | (label_bridge & close_to_topic)
    num_pos = int(pos_mask.sum().item())
    num_bridge = int(label_bridge.sum().item())
    if num_pos < min(num_bridge, config['pairwise']['max_pos']):
        bridge_candidates = torch.nonzero(label_bridge, as_tuple=False).reshape(-1)
        if bridge_candidates.numel() > 0:
            bridge_candidate_set = set(bridge_candidates.tolist())
            rank_order = torch.argsort(sample['f_stage1'][:, 2], descending=True)
            rank_priority = [
                idx.item() for idx in rank_order
                if idx.item() in bridge_candidate_set
            ]
            needed = min(config['pairwise']['max_pos'], len(rank_priority))
            if needed > 0:
                pos_mask[torch.tensor(rank_priority[:needed], device=labels.device)] = True

    hard_negative_k = min(
        config['pairwise']['hard_negative_top_k'],
        max(1, int(num_candidates * config['pairwise']['hard_negative_top_percent']))
    )
    rank_order = torch.argsort(sample['f_stage1'][:, 2], descending=True)
    top_stage1 = rank_order[:hard_negative_k]
    neg_mask = label_zero.clone()
    keep_neg = torch.zeros_like(neg_mask, dtype=torch.bool)
    keep_neg[top_stage1] = True
    neg_mask = neg_mask & keep_neg

    max_neg = int(config['pairwise']['max_neg'])
    pw = config['pairwise']
    if pw.get('mid_band_enabled', True):
        mid_lo = min(int(pw.get('mid_band_start_idx', 49)), num_candidates - 1)
        mid_hi = min(int(pw.get('mid_band_end_idx', 149)), num_candidates - 1)
        mid_mask = torch.zeros(num_candidates, dtype=torch.bool, device=labels.device)
        if mid_lo <= mid_hi:
            mid_mask[mid_lo:mid_hi + 1] = True
        neg_mid = torch.nonzero(label_zero & mid_mask, as_tuple=False).reshape(-1)
        neg_top = torch.nonzero(neg_mask, as_tuple=False).reshape(-1)
        priority = []
        seen = set()
        for tensor in (neg_mid, neg_top):
            for idx in tensor.tolist():
                if idx in seen:
                    continue
                seen.add(idx)
                priority.append(idx)
                if len(priority) >= max_neg:
                    break
            if len(priority) >= max_neg:
                break
        if priority:
            neg_indices = torch.tensor(priority, device=labels.device, dtype=torch.long)
        else:
            neg_indices = torch.nonzero(neg_mask, as_tuple=False).reshape(-1)
            if neg_indices.numel() > max_neg:
                neg_indices = neg_indices[:max_neg]
    else:
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).reshape(-1)
        if neg_indices.numel() > max_neg:
            neg_indices = neg_indices[:max_neg]

    pos_indices = torch.nonzero(pos_mask, as_tuple=False).reshape(-1)
    if pos_indices.numel() > config['pairwise']['max_pos']:
        pos_indices = pos_indices[:config['pairwise']['max_pos']]

    return pos_indices, neg_indices


def pairwise_logistic_loss(logits: torch.Tensor, sample: dict, config: dict):
    if not config['pairwise']['enabled']:
        return logits.new_tensor(0.0)

    pos_indices, neg_indices = select_pairwise_examples(sample, config)
    if pos_indices.numel() == 0 or neg_indices.numel() == 0:
        return logits.new_tensor(0.0)

    pos_logits = logits[pos_indices]
    neg_logits = logits[neg_indices]
    margin = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
    return torch.log1p(torch.exp(-margin)).mean()
