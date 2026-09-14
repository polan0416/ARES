import json
import math
import os
import sys
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config.stage2_reranker import get_label_config
from src.dataset.retriever import RetrieverDataset
from src.model.retriever import Retriever


RESULT_FILE_BY_SPLIT = {
    'train': 'retrieval_result_train.pth',
    'val': 'retrieval_result_val.pth',
    'test': 'retrieval_result.pth',
}

SILVER_FILE_BY_SPLIT = {
    'train': 'train_silver_subgraphs.json',
    'val': 'val_silver_subgraphs.json',
    'test': 'test_silver_subgraphs.json',
}

GOLD_FILE_BY_SPLIT = {
    'train': 'train_gold_subgraphs.json',
    'val': 'val_gold_subgraphs.json',
    'test': 'test_gold_subgraphs.json',
}


def _normalize_value(value):
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_triple(triple: Sequence) -> Tuple:
    return tuple(_normalize_value(value) for value in triple[:3])


def resolve_result_file(retrieval_dir: str, split: str) -> str:
    if split not in RESULT_FILE_BY_SPLIT:
        raise ValueError(f'Unsupported split: {split}')
    primary = os.path.join(retrieval_dir, RESULT_FILE_BY_SPLIT[split])
    if os.path.exists(primary):
        return primary
    if split == 'test':
        alt = os.path.join(retrieval_dir, 'retrieval_result_test.pth')
        if os.path.exists(alt):
            return alt
    return primary


def resolve_silver_file(silver_dir: str, split: str) -> str:
    if split not in SILVER_FILE_BY_SPLIT:
        raise ValueError(f'Unsupported split: {split}')

    silver_path = os.path.join(silver_dir, SILVER_FILE_BY_SPLIT[split])
    if os.path.exists(silver_path):
        return silver_path

    gold_path = os.path.join(silver_dir, GOLD_FILE_BY_SPLIT[split])
    return gold_path


def load_retrieval_results(retrieval_dir: str, split: str) -> Dict[str, Dict]:
    file_path = resolve_result_file(retrieval_dir, split)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Missing retrieval result file: {file_path}')
    return torch.load(file_path, map_location='cpu', weights_only=False)


def load_silver_triplets(silver_dir: str, split: str) -> Dict[str, Dict]:
    file_path = resolve_silver_file(silver_dir, split)
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_tensor_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == 'float16':
        return torch.float16
    if dtype_name == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported tensor dtype: {dtype_name}')


def stage1_rank_norm(num_candidates: int) -> torch.Tensor:
    if num_candidates <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if num_candidates == 1:
        return torch.ones(1, dtype=torch.float32)
    ranks = torch.arange(num_candidates, dtype=torch.float32)
    return 1.0 - ranks / float(num_candidates - 1)


def get_topic_entities(sample: Dict) -> List[str]:
    values = sample.get('q_entity_in_graph') or sample.get('q_entity') or []
    return [_normalize_value(value) for value in values]


def get_answer_entities(sample: Dict) -> List[str]:
    values = sample.get('a_entity_in_graph') or sample.get('a_entity') or []
    return [_normalize_value(value) for value in values]


def build_raw_triple_lookup(raw_sample: Dict):
    entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
    relation_list = raw_sample['relation_list']

    raw_triples = []
    triple_lookup = defaultdict(deque)
    duplicate_count = 0

    for idx, (h_id, r_id, t_id) in enumerate(zip(
        raw_sample['h_id_list'],
        raw_sample['r_id_list'],
        raw_sample['t_id_list'],
    )):
        triple = (
            entity_list[h_id],
            relation_list[r_id],
            entity_list[t_id],
        )
        raw_triples.append(triple)
        key = normalize_triple(triple)
        if triple_lookup[key]:
            duplicate_count += 1
        triple_lookup[key].append(idx)

    return raw_triples, triple_lookup, duplicate_count


def align_result_sample_to_raw_indices(
    raw_sample: Dict,
    result_sample: Dict,
    top_k: int,
):
    raw_triples, triple_lookup, duplicate_count = build_raw_triple_lookup(raw_sample)
    aligned_indices = []
    alignment_conflicts = 0

    for result_triple in result_sample.get('scored_triples', [])[:top_k]:
        key = normalize_triple(result_triple)
        candidate_queue = triple_lookup.get(key)
        if not candidate_queue:
            raise KeyError(
                f'Cannot align retrieval triple {key} for sample {raw_sample["id"]}')
        if len(candidate_queue) > 1:
            alignment_conflicts += len(candidate_queue) - 1
        aligned_indices.append(candidate_queue.popleft())

    return aligned_indices, {
        'num_raw_duplicate_triples': duplicate_count,
        'alignment_conflicts': alignment_conflicts,
        'num_candidates': len(aligned_indices),
    }


def build_candidate_graph(candidate_triples: Sequence[Sequence]):
    adjacency = defaultdict(set)
    reverse_adjacency = defaultdict(set)
    pair_to_indices = defaultdict(list)
    entities = set()

    for idx, triple in enumerate(candidate_triples):
        h, _, t = normalize_triple(triple)
        adjacency[h].add(t)
        reverse_adjacency[t].add(h)
        pair_to_indices[(h, t)].append(idx)
        entities.add(h)
        entities.add(t)

    return adjacency, reverse_adjacency, pair_to_indices, entities


def multi_source_bfs(
    adjacency: Dict,
    sources: Iterable,
    max_depth: Optional[int] = None,
) -> Dict:
    distances = {}
    queue = deque()

    for source in sources:
        if source in distances:
            continue
        distances[source] = 0
        queue.append(source)

    while queue:
        node = queue.popleft()
        if max_depth is not None and distances[node] >= max_depth:
            continue
        for neighbor in adjacency.get(node, []):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)

    return distances


def compute_topic_dir_distances(
    candidate_triples: Sequence[Sequence],
    topic_entities: Sequence[str],
    distance_cap: int,
) -> torch.Tensor:
    if len(candidate_triples) == 0:
        return torch.empty((0, 4), dtype=torch.long)

    adjacency, reverse_adjacency, _, _ = build_candidate_graph(candidate_triples)
    topic_entities = [_normalize_value(entity) for entity in topic_entities]

    topic_to_dist = multi_source_bfs(adjacency, topic_entities)
    to_topic_dist = multi_source_bfs(reverse_adjacency, topic_entities)

    raw_distances = []
    for triple in candidate_triples:
        h, _, t = normalize_triple(triple)
        raw_distances.append([
            min(topic_to_dist.get(h, distance_cap), distance_cap),
            min(topic_to_dist.get(t, distance_cap), distance_cap),
            min(to_topic_dist.get(h, distance_cap), distance_cap),
            min(to_topic_dist.get(t, distance_cap), distance_cap),
        ])

    return torch.tensor(raw_distances, dtype=torch.long)


def shortest_path_edge_indices_for_pair(
    adjacency: Dict,
    reverse_adjacency: Dict,
    pair_to_indices: Dict,
    source: str,
    target: str,
    forward_cache: Dict,
    reverse_cache: Dict,
):
    if source not in forward_cache:
        forward_cache[source] = multi_source_bfs(adjacency, [source])
    if target not in reverse_cache:
        reverse_cache[target] = multi_source_bfs(reverse_adjacency, [target])

    source_dist = forward_cache[source]
    target_dist = reverse_cache[target]
    if target not in source_dist:
        return None, set()

    path_length = source_dist[target]
    edge_indices = set()
    for (head, tail), triple_indices in pair_to_indices.items():
        dist_head = source_dist.get(head)
        dist_tail = target_dist.get(tail)
        if dist_head is None or dist_tail is None:
            continue
        if dist_head + 1 + dist_tail == path_length:
            edge_indices.update(triple_indices)

    return path_length, edge_indices


def collect_shortest_path_triples(
    candidate_triples: Sequence[Sequence],
    topic_entities: Sequence[str],
    answer_entities: Sequence[str],
):
    adjacency, reverse_adjacency, pair_to_indices, entities = build_candidate_graph(candidate_triples)
    normalized_topics = [_normalize_value(entity) for entity in topic_entities if _normalize_value(entity) in entities]
    normalized_answers = [_normalize_value(entity) for entity in answer_entities if _normalize_value(entity) in entities]

    forward_cache = {}
    reverse_cache = {}
    shortest_indices = set()

    for topic_entity in normalized_topics:
        for answer_entity in normalized_answers:
            forward_len, forward_edges = shortest_path_edge_indices_for_pair(
                adjacency,
                reverse_adjacency,
                pair_to_indices,
                topic_entity,
                answer_entity,
                forward_cache,
                reverse_cache,
            )
            backward_len, backward_edges = shortest_path_edge_indices_for_pair(
                adjacency,
                reverse_adjacency,
                pair_to_indices,
                answer_entity,
                topic_entity,
                forward_cache,
                reverse_cache,
            )

            valid_lengths = [length for length in [forward_len, backward_len] if length is not None]
            if not valid_lengths:
                continue
            min_length = min(valid_lengths)
            if forward_len is not None and forward_len == min_length:
                shortest_indices.update(forward_edges)
            if backward_len is not None and backward_len == min_length:
                shortest_indices.update(backward_edges)

    shortest_triples = [normalize_triple(candidate_triples[idx]) for idx in sorted(shortest_indices)]
    return shortest_indices, shortest_triples


def match_gold_or_silver_triples(
    candidate_triples: Sequence[Sequence],
    supervision_sample: Optional[Dict],
):
    triplets = supervision_sample.get('triplets', []) if supervision_sample else []

    direct_keys = set()
    bridge_keys = set()
    weak_keys = set()

    for item in triplets:
        if isinstance(item, dict):
            head = item.get('head')
            rel = item.get('relation')
            tail = item.get('tail')
            label = item.get('label')
            if head is None or rel is None or tail is None or not label:
                continue
            key = normalize_triple((head, rel, tail))
            if label == 'direct_support':
                direct_keys.add(key)
            elif label == 'bridge':
                bridge_keys.add(key)
            elif label == 'weak_support':
                weak_keys.add(key)
        else:
            key = normalize_triple(item)
            bridge_keys.add(key)

    candidate_keys = [normalize_triple(triple) for triple in candidate_triples]
    direct_indices = {i for i, key in enumerate(candidate_keys) if key in direct_keys}
    bridge_indices = {i for i, key in enumerate(candidate_keys) if key in bridge_keys}
    weak_indices = {i for i, key in enumerate(candidate_keys) if key in weak_keys}

    direct_triples = [candidate_keys[i] for i in sorted(direct_indices)]
    bridge_triples = [candidate_keys[i] for i in sorted(bridge_indices)]
    weak_triples = [candidate_keys[i] for i in sorted(weak_indices)]
    return direct_indices, bridge_indices, weak_indices, direct_triples, bridge_triples, weak_triples


def build_soft_labels(
    num_candidates: int,
    shortest_indices: set,
    direct_indices: set,
    bridge_indices: set,
    weak_indices: set,
    direct_label: float,
    bridge_label: float,
    weak_label: float,
):
    soft_labels = torch.zeros(num_candidates, dtype=torch.float32)

    labeled_indices = set()

    if direct_indices:
        labeled_indices.update(direct_indices)
        soft_labels[list(sorted(direct_indices))] = float(direct_label)

    if bridge_indices:
        labeled_indices.update(bridge_indices)
        soft_labels[list(sorted(bridge_indices))] = bridge_label

    if weak_indices:
        labeled_indices.update(weak_indices)
        soft_labels[list(sorted(weak_indices))] = weak_label

    if shortest_indices:
        unlabeled_shortest = set(shortest_indices) - labeled_indices
        if unlabeled_shortest:
            soft_labels[list(sorted(unlabeled_shortest))] = 1.0
    return soft_labels


def compute_structural_features(candidate_triples: Sequence[Sequence]) -> torch.Tensor:
    if len(candidate_triples) == 0:
        return torch.empty(0, 8, dtype=torch.float32)

    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    entity_mention = defaultdict(int)
    r_counts = defaultdict(int)
    pair_counts = defaultdict(int)

    for triple in candidate_triples:
        h, r, t = normalize_triple(triple)
        r_key = str(r).strip()
        out_deg[h] += 1
        in_deg[t] += 1
        entity_mention[h] += 1
        entity_mention[t] += 1
        r_counts[r_key] += 1
        pair_counts[(h, t)] += 1

    max_out = max(out_deg.values()) if out_deg else 1
    max_in = max(in_deg.values()) if in_deg else 1
    max_m = max(entity_mention.values()) if entity_mention else 1
    max_r = max(r_counts.values()) if r_counts else 1

    def log_norm(count: int, max_count: int) -> float:
        return math.log(1 + count) / math.log(1 + max(max_count, 1))

    rows = []
    for triple in candidate_triples:
        h, r, t = normalize_triple(triple)
        r_key = str(r).strip()
        rows.append([
            log_norm(out_deg[h], max_out),
            log_norm(in_deg[h], max_in),
            log_norm(out_deg[t], max_out),
            log_norm(in_deg[t], max_in),
            log_norm(entity_mention[h], max_m),
            log_norm(entity_mention[t], max_m),
            log_norm(r_counts[r_key], max_r),
            1.0 if pair_counts[(h, t)] > 1 else 0.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def build_stage2_inputs(sample: Dict) -> torch.Tensor:
    required = (
        'x_stage1',
        'f_struct',
    )
    for key in required:
        if key not in sample:
            raise KeyError(
                f'Stage2 sample missing {key}. Rebuild cache (stage2_feature_version>=4).'
            )
    return torch.cat([
        sample['x_stage1'].float(),
        sample['f_struct'].float(),
    ], dim=-1)


def prepare_stage2_sample(device: torch.device, sample: Dict) -> Dict:
    prepared = dict(sample)
    for key, value in sample.items():
        if torch.is_tensor(value):
            prepared[key] = value.to(device)
    prepared['x_stage2'] = build_stage2_inputs(prepared)
    return prepared


class Stage2CacheDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str,
        filter_all_negative: Optional[bool] = None,
        max_samples: Optional[int] = None,
        oversample_multi_answer: bool = False,
        multi_answer_extra_copies: int = 1,
    ):
        manifest_path = os.path.join(cache_dir, f'{split}_manifest.pt')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'Missing stage2 manifest: {manifest_path}')

        manifest = torch.load(manifest_path, map_location='cpu', weights_only=False)
        entries = manifest['entries']
        if filter_all_negative is None:
            filter_all_negative = split == 'train'
        if filter_all_negative:
            entries = [entry for entry in entries if entry['has_positive']]
        if max_samples is not None:
            entries = entries[:max_samples]

        self.cache_dir = cache_dir
        self.split = split
        self.entries = entries
        self.manifest = manifest
        self._indices = self._build_epoch_indices(
            oversample_multi_answer=oversample_multi_answer and split == 'train',
            multi_answer_extra_copies=max(0, int(multi_answer_extra_copies)),
        )

    def _build_epoch_indices(
        self,
        oversample_multi_answer: bool,
        multi_answer_extra_copies: int,
    ):
        n = len(self.entries)
        base = list(range(n))
        if not oversample_multi_answer or multi_answer_extra_copies <= 0:
            return base
        out = []
        for i in range(n):
            out.append(i)
            na = int(self.entries[i].get('num_answers', 1))
            if na >= 2:
                out.extend([i] * multi_answer_extra_copies)
        return out

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        entry = self.entries[self._indices[index]]
        sample = torch.load(entry['file_path'], map_location='cpu', weights_only=False)
        sample['num_answers'] = int(entry.get('num_answers', 1))
        return sample


def collate_stage2_reranker(data):
    return data[0]


class Stage2SampleBuilder:
    def __init__(self, config: Dict, split: str, device: torch.device):
        self.config = config
        self.split = split
        self.device = device

        stage1_checkpoint = config['stage1']['checkpoint_path']
        cpt = torch.load(stage1_checkpoint, map_location='cpu', weights_only=False)
        self.stage1_config = cpt['config']

        self.dataset = RetrieverDataset(
            config=self.stage1_config,
            split=split,
            skip_no_path=False,
        )
        self.sample_by_id = {
            sample['id']: sample for sample in self.dataset.processed_dict_list
        }

        self.result_dict = load_retrieval_results(config['stage1']['retrieval_dir'], split)
        label_cfg = get_label_config(config)
        self.silver_dict = load_silver_triplets(label_cfg['dir'], split)

        sample0 = self.dataset[0]
        emb_size = sample0['q_emb'].shape[-1]
        self.model = Retriever(emb_size, **self.stage1_config['retriever']).to(device)
        self.model.load_state_dict(cpt['model_state_dict'])
        self.model.eval()

        self.top_k = config['stage1']['candidate_top_k']
        self.distance_cap = config['cache']['distance_cap']
        self.bridge_label = float(
            label_cfg.get('bridge_label', label_cfg.get('silver_label', 0.7))
        )
        self.direct_label = float(label_cfg.get('direct_label', 1.0))
        self.weak_label = float(label_cfg.get('weak_label', 0.1))
        self.tensor_dtype = resolve_tensor_dtype(config['cache']['tensor_dtype'])

    @torch.no_grad()
    def build_sample(self, sample_id: str) -> Dict:
        raw_sample = self.sample_by_id[sample_id]
        result_sample = self.result_dict[sample_id]
        silver_sample = self.silver_dict.get(sample_id)

        candidate_entries = result_sample.get('scored_triples', [])[:self.top_k]
        raw_indices, align_stats = align_result_sample_to_raw_indices(
            raw_sample=raw_sample,
            result_sample=result_sample,
            top_k=self.top_k,
        )

        selected_h = torch.tensor(
            [raw_sample['h_id_list'][idx] for idx in raw_indices],
            dtype=torch.long,
            device=self.device,
        )
        selected_r = torch.tensor(
            [raw_sample['r_id_list'][idx] for idx in raw_indices],
            dtype=torch.long,
            device=self.device,
        )
        selected_t = torch.tensor(
            [raw_sample['t_id_list'][idx] for idx in raw_indices],
            dtype=torch.long,
            device=self.device,
        )
        full_h = torch.tensor(raw_sample['h_id_list'], dtype=torch.long, device=self.device)
        full_t = torch.tensor(raw_sample['t_id_list'], dtype=torch.long, device=self.device)

        stage1_features = self.model.forward_with_features(
            h_id_tensor=selected_h,
            r_id_tensor=selected_r,
            t_id_tensor=selected_t,
            q_emb=raw_sample['q_emb'].to(self.device),
            entity_embs=raw_sample['entity_embs'].to(self.device),
            num_non_text_entities=len(raw_sample['non_text_entity_list']),
            relation_embs=raw_sample['relation_embs'].to(self.device),
            topic_entity_one_hot=raw_sample['topic_entity_one_hot'].to(self.device),
            graph_h_id_tensor=full_h,
            graph_t_id_tensor=full_t,
        )

        stage1_prob = stage1_features['stage1_prob'].reshape(-1).detach().cpu()
        stage1_logit = stage1_features['stage1_logit'].reshape(-1).detach().cpu()
        x_stage1 = stage1_features['x_stage1'].detach().cpu()
        rank_norm = stage1_rank_norm(len(candidate_entries))
        f_stage1 = torch.stack([stage1_prob, stage1_logit, rank_norm], dim=-1)
        f_struct = compute_structural_features(candidate_entries)

        topic_entities = get_topic_entities(result_sample)
        answer_entities = get_answer_entities(result_sample)
        topic_dir_distances = compute_topic_dir_distances(
            candidate_triples=candidate_entries,
            topic_entities=topic_entities,
            distance_cap=self.distance_cap,
        )
        shortest_indices, shortest_triples = collect_shortest_path_triples(
            candidate_triples=candidate_entries,
            topic_entities=topic_entities,
            answer_entities=answer_entities,
        )
        direct_indices, bridge_indices, weak_indices, direct_triples, bridge_triples, weak_triples = match_gold_or_silver_triples(
            candidate_triples=candidate_entries,
            supervision_sample=silver_sample,
        )
        soft_labels = build_soft_labels(
            num_candidates=len(candidate_entries),
            shortest_indices=shortest_indices,
            direct_indices=direct_indices,
            bridge_indices=bridge_indices,
            weak_indices=weak_indices,
            direct_label=self.direct_label,
            bridge_label=self.bridge_label,
            weak_label=self.weak_label,
        )

        result_scores = []
        for triple in candidate_entries:
            if len(triple) >= 5:
                result_scores.append(float(triple[4]))
            else:
                result_scores.append(float('nan'))
        result_scores = torch.tensor(result_scores, dtype=torch.float32)
        valid_result_scores = ~torch.isnan(result_scores)
        if valid_result_scores.any():
            stage1_prob_max_abs_diff = float(
                (stage1_prob[valid_result_scores] - result_scores[valid_result_scores]).abs().max())
        else:
            stage1_prob_max_abs_diff = math.nan

        cache_sample = {
            'id': sample_id,
            'split': self.split,
            'stage2_feature_version': 4,
            'question': result_sample.get('question', raw_sample['question']),
            'q_entity': list(result_sample.get('q_entity', raw_sample['q_entity'])),
            'q_entity_in_graph': list(result_sample.get('q_entity_in_graph', topic_entities)),
            'a_entity': list(result_sample.get('a_entity', raw_sample['a_entity'])),
            'a_entity_in_graph': list(result_sample.get('a_entity_in_graph', answer_entities)),
            'max_path_length': result_sample.get('max_path_length', raw_sample.get('max_path_length')),
            'target_relevant_triples': list(result_sample.get('target_relevant_triples', [])),
            'scored_triples': list(candidate_entries),
            'candidate_triples_norm': [normalize_triple(triple) for triple in candidate_entries],
            'raw_candidate_indices': torch.tensor(raw_indices, dtype=torch.long),
            'x_stage1': x_stage1.to(self.tensor_dtype),
            'f_stage1': f_stage1.to(self.tensor_dtype),
            'f_struct': f_struct.to(self.tensor_dtype),
            'soft_labels': soft_labels.to(torch.float32),
            'topic_dir_distances': topic_dir_distances,
            'shortest_path_indices': torch.tensor(sorted(shortest_indices), dtype=torch.long),
            'silver_indices': torch.tensor(sorted(bridge_indices), dtype=torch.long),
            'shortest_path_triples': shortest_triples,
            'silver_triples': bridge_triples,
            'direct_triples': direct_triples,
            'weak_triples': weak_triples,
            'alignment_stats': align_stats,
            'stage1_prob_max_abs_diff': stage1_prob_max_abs_diff,
        }
        return cache_sample

    def count_samples(self, max_samples: Optional[int] = None) -> int:
        sample_ids = [sample['id'] for sample in self.dataset.processed_dict_list]
        if max_samples is not None:
            sample_ids = sample_ids[:max_samples]
        return sum(1 for sample_id in sample_ids if sample_id in self.result_dict)

    def iter_samples(self, max_samples: Optional[int] = None):
        sample_ids = [sample['id'] for sample in self.dataset.processed_dict_list]
        if max_samples is not None:
            sample_ids = sample_ids[:max_samples]
        for sample_id in sample_ids:
            if sample_id not in self.result_dict:
                continue
            yield self.build_sample(sample_id)

    def build_cache(self, cache_dir: str, max_samples: Optional[int] = None):
        os.makedirs(cache_dir, exist_ok=True)
        entries = []
        total_positive = 0
        total_shortest = 0
        num_samples = self.count_samples(max_samples=max_samples)

        for sample in tqdm(
            self.iter_samples(max_samples=max_samples),
            total=num_samples,
            desc=f'Build stage2 cache [{self.split}]',
            unit='sample',
            dynamic_ncols=True,
            mininterval=0.2,
            file=sys.stderr,
        ):
            file_path = os.path.join(cache_dir, f'{sample["id"]}.pt')
            torch.save(sample, file_path)
            num_positive = int((sample['soft_labels'] > 0).sum().item())
            num_shortest = int((sample['soft_labels'] == 1.0).sum().item())
            total_positive += num_positive
            total_shortest += num_shortest
            ans_list = sample.get('a_entity_in_graph') or sample.get('a_entity') or []
            num_answers = max(1, sum(1 for x in ans_list if str(x).strip()))
            entries.append({
                'sample_id': sample['id'],
                'file_path': file_path,
                'num_candidates': int(sample['soft_labels'].numel()),
                'num_positive': num_positive,
                'num_shortest': num_shortest,
                'has_positive': bool(num_positive > 0),
                'num_answers': num_answers,
                'stage1_prob_max_abs_diff': sample['stage1_prob_max_abs_diff'],
            })

        manifest = {
            'split': self.split,
            'entries': entries,
            'num_samples': len(entries),
            'total_positive': total_positive,
            'total_shortest': total_shortest,
        }
        manifest_path = os.path.join(os.path.dirname(cache_dir), f'{self.split}_manifest.pt')
        torch.save(manifest, manifest_path)
        return manifest
