import os
from collections import deque

import torch

from tqdm import tqdm

from src.setup import set_seed, prepare_sample


RESULT_FILE_NAMES = (
    "retrieval_result_train.pth",
    "retrieval_result_val.pth",
    "retrieval_result.pth",
)


def _extract_topic_entities(sample):
    topic_entities = sample.get("q_entity_in_graph") or sample.get("q_entity") or []
    return set(topic_entities)


def _normalize_triple(triple):
    return triple[:3]


def _compute_entity_distances(triples, topic_entities):
    adjacency = {}
    for h, _, t in triples:
        adjacency.setdefault(h, set()).add(t)
        adjacency.setdefault(t, set()).add(h)

    queue = deque()
    distances = {}
    for entity in topic_entities:
        if entity in distances:
            continue
        distances[entity] = 0
        queue.append(entity)

    while queue:
        entity = queue.popleft()
        for neighbor in adjacency.get(entity, set()):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[entity] + 1
            queue.append(neighbor)

    return distances


def _annotate_triples_with_hop(triples, topic_entities):
    normalized_triples = [_normalize_triple(triple) for triple in triples]
    distances = _compute_entity_distances(normalized_triples, topic_entities)

    annotated_triples = []
    for idx, (h, r, t) in enumerate(normalized_triples):
        endpoint_distances = [distances[entity] for entity in (h, t) if entity in distances]
        hop = min(endpoint_distances) + 1 if endpoint_distances else None
        score = None
        if idx < len(triples):
            original = triples[idx]
            if isinstance(original, (list, tuple)) and len(original) >= 4:
                score = original[3]
        annotated_triples.append((h, r, t, hop, score))

    return annotated_triples


def rewrite_pred_dict_with_hop(pred_dict):
    rewritten_pred_dict = {}
    for sample_id, sample in pred_dict.items():
        rewritten_sample = sample.copy()
        topic_entities = _extract_topic_entities(sample)
        rewritten_sample["scored_triples"] = _annotate_triples_with_hop(
            sample.get("scored_triples", []),
            topic_entities,
        )
        rewritten_sample["target_relevant_triples"] = _annotate_triples_with_hop(
            sample.get("target_relevant_triples", []),
            topic_entities,
        )
        rewritten_pred_dict[sample_id] = rewritten_sample

    return rewritten_pred_dict


def rewrite_result_file(file_path):
    pred_dict = torch.load(file_path, map_location="cpu")
    rewritten_pred_dict = rewrite_pred_dict_with_hop(pred_dict)
    torch.save(rewritten_pred_dict, file_path)
    print(f"Rewritten {len(rewritten_pred_dict)} samples in {file_path}")


def rewrite_result_dir(result_dir):
    for file_name in RESULT_FILE_NAMES:
        file_path = os.path.join(result_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing retrieval result file: {file_path}")
        rewrite_result_file(file_path)


def run_inference_for_split(device, model, config, split, args):
    from src.dataset.retriever import RetrieverDataset, collate_retriever

    infer_set = RetrieverDataset(
        config=config, split=split, skip_no_path=False)
    pred_dict = dict()
    for i in tqdm(range(len(infer_set)), desc=f'Inference {split}'):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            sorted_indices = torch.argsort(pred_triple_scores, descending=True)
            seen_triples = set()
            for idx in sorted_indices.cpu().tolist():
                if len(top_K_triples) >= args.max_K:
                    break
                triple_id = int(idx)
                h = entity_list[h_id_tensor[triple_id].item()]
                r = relation_list[r_id_tensor[triple_id].item()]
                t = entity_list[t_id_tensor[triple_id].item()]
                key = (h, r, t)
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                score = float(pred_triple_scores[triple_id].item())
                top_K_triples.append((h, r, t, score))

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }

        pred_dict[raw_sample['id']] = sample_dict
    return rewrite_pred_dict_with_hop(pred_dict)


@torch.no_grad()
def main(args):
    if args.rewrite_dir is not None:
        rewrite_result_dir(args.rewrite_dir)
        return

    if args.path is None:
        raise ValueError("Either --path or --rewrite_dir must be provided.")

    device = torch.device(f'cuda:0')

    from src.dataset.retriever import RetrieverDataset
    from src.model.retriever import Retriever

    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    if args.split == 'all':
        splits_to_run = ['train', 'val', 'test']
    else:
        splits_to_run = [args.split]

    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    root_path = os.path.dirname(args.path)
    for split in splits_to_run:
        pred_dict = run_inference_for_split(device, model, config, split, args)
        out_name = 'retrieval_result.pth' if split == 'test' else f'retrieval_result_{split}.pth'
        out_path = os.path.join(root_path, out_name)
        torch.save(pred_dict, out_path)
        print(f'Saved {len(pred_dict)} samples to {out_path}')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--rewrite_dir', type=str,
                        help='Directory containing retrieval_result_train.pth, retrieval_result_val.pth, and retrieval_result.pth')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Data split: train, val, test, or all (train+val+test). Default: test')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    args = parser.parse_args()

    main(args)
