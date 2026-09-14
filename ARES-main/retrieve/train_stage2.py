import os
import time
from collections import defaultdict

import sys
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.stage2_reranker import load_yaml, resolve_config_path
from src.dataset.stage2_reranker import (
    Stage2CacheDataset,
    build_stage2_inputs,
    collate_stage2_reranker,
    prepare_stage2_sample,
)
from src.model.stage2_reranker import (
    Stage2TripleReranker,
    answer_touch_bce_loss,
    compute_inference_fusion_score,
    min_answer_coverage_loss,
    pairwise_logistic_loss,
    soft_bce_loss,
    topk_listwise_surrogate_loss,
)
from src.setup import set_seed


@torch.no_grad()
def eval_epoch(config, device, data_loader, model):
    model.eval()
    metrics = defaultdict(list)
    pairwise_weight = config['pairwise']['weight'] if config['pairwise']['enabled'] else 0.0
    aux_w = float(config.get('answer_aux', {}).get('weight', 0.0))
    min_aw = float(config.get('answer_aux', {}).get('min_answer_weight', 0.0))
    bucket_enable = bool(config.get('answer_aux', {}).get('bucket_enable', False))
    n_ans1_touch_mult = float(config.get('answer_aux', {}).get('n_ans1_touch_mult', 1.0))
    n_ans_ge2_touch_mult = float(config.get('answer_aux', {}).get('n_ans_ge2_touch_mult', 1.0))
    n_ans1_min_cov_mult = float(config.get('answer_aux', {}).get('n_ans1_min_cov_mult', 1.0))
    n_ans_ge2_min_cov_mult = float(config.get('answer_aux', {}).get('n_ans_ge2_min_cov_mult', 1.0))
    topk_w = float(config.get('answer_aux', {}).get('topk_listwise_weight', 0.0))
    aux_enabled = config.get('answer_aux', {}).get('enabled', False)

    for sample in tqdm(
        data_loader,
        desc='Eval',
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
        file=sys.stderr,
    ):
        sample = prepare_stage2_sample(device, sample)
        logits = model(sample['x_stage2'])
        bce = soft_bce_loss(logits, sample['soft_labels'])
        pair = pairwise_logistic_loss(logits, sample, config)
        ans = answer_touch_bce_loss(logits, sample, config)
        min_ans = min_answer_coverage_loss(logits, sample, config)
        topk_ls = topk_listwise_surrogate_loss(logits, sample, config)
        loss = bce + pairwise_weight * pair
        if aux_enabled:
            n_ans = int(sample.get('num_answers', 1))
            touch_mult = 1.0
            min_cov_mult = 1.0
            if bucket_enable:
                touch_mult = n_ans1_touch_mult if n_ans <= 1 else n_ans_ge2_touch_mult
                min_cov_mult = n_ans1_min_cov_mult if n_ans <= 1 else n_ans_ge2_min_cov_mult
            loss = loss + aux_w * touch_mult * ans
            if min_aw > 0:
                loss = loss + min_aw * min_cov_mult * min_ans
            if topk_w > 0:
                loss = loss + topk_w * topk_ls

        metrics['loss'].append(loss.item())
        metrics['bce_loss'].append(bce.item())
        metrics['pairwise_loss'].append(pair.item())
        metrics['answer_aux_loss'].append(ans.item())
        metrics['min_answer_loss'].append(min_ans.item())
        metrics['topk_listwise_loss'].append(topk_ls.item())

        if logits.numel() == 0:
            continue

        score_final = compute_inference_fusion_score(sample, logits, config)
        stage2_probs = torch.sigmoid(logits).to(dtype=torch.float32)
        stage1_probs = sample['f_stage1'][:, 0].to(dtype=torch.float32)
        sorted_indices = torch.argsort(score_final, descending=True)
        candidate_ranks = torch.empty_like(sorted_indices)
        candidate_ranks[sorted_indices] = torch.arange(len(sorted_indices), device=sorted_indices.device)

        positive_indices = torch.nonzero(sample['soft_labels'] > 0, as_tuple=False).reshape(-1)
        shortest_indices = sample['shortest_path_indices']
        max_k = min(max(config['eval']['k_list']), logits.numel())
        top_indices = sorted_indices[:max_k]
        metrics[f'mean_label@{max_k}'].append(sample['soft_labels'][top_indices].float().mean().item())

        answer_entities = sample.get('a_entity_in_graph') or sample.get('a_entity') or []
        answer_entities = [str(x).strip() for x in answer_entities]
        answer_entity_set = set(answer_entities)
        candidate_triples_norm = sample.get('candidate_triples_norm') or []

        for k in config['eval']['k_list']:
            top_k = min(k, logits.numel())
            if top_k == 0:
                continue
            if positive_indices.numel() > 0:
                recall = (candidate_ranks[positive_indices] < top_k).float().mean().item()
                metrics[f'positive_recall@{k}'].append(recall)
            if shortest_indices.numel() > 0:
                recall = (candidate_ranks[shortest_indices] < top_k).float().mean().item()
                metrics[f'shortest_recall@{k}'].append(recall)

            if answer_entity_set and candidate_triples_norm:
                pred_topk = sorted_indices[:top_k].detach().cpu().tolist()
                hit_entities = set()
                entities_in_topk = set()
                for idx in pred_topk:
                    if idx < 0 or idx >= len(candidate_triples_norm):
                        continue
                    h, _, t = candidate_triples_norm[idx]
                    h = str(h).strip()
                    t = str(t).strip()
                    entities_in_topk.add(h)
                    entities_in_topk.add(t)
                    if h in answer_entity_set:
                        hit_entities.add(h)
                    if t in answer_entity_set:
                        hit_entities.add(t)
                metrics[f'answer_recall@{k}'].append(float(len(hit_entities) / len(answer_entity_set)))
                metrics[f'full_cov@{k}'].append(
                    1.0 if answer_entity_set.issubset(entities_in_topk) else 0.0
                )

    reduced = {}
    for key, values in metrics.items():
        reduced[key] = float(sum(values) / len(values)) if values else 0.0
    return reduced


def train_epoch(config, device, data_loader, model, optimizer):
    model.train()
    metrics = defaultdict(list)
    pairwise_weight = config['pairwise']['weight'] if config['pairwise']['enabled'] else 0.0
    aux_w = float(config.get('answer_aux', {}).get('weight', 0.0))
    min_aw = float(config.get('answer_aux', {}).get('min_answer_weight', 0.0))
    bucket_enable = bool(config.get('answer_aux', {}).get('bucket_enable', False))
    n_ans1_touch_mult = float(config.get('answer_aux', {}).get('n_ans1_touch_mult', 1.0))
    n_ans_ge2_touch_mult = float(config.get('answer_aux', {}).get('n_ans_ge2_touch_mult', 1.0))
    n_ans1_min_cov_mult = float(config.get('answer_aux', {}).get('n_ans1_min_cov_mult', 1.0))
    n_ans_ge2_min_cov_mult = float(config.get('answer_aux', {}).get('n_ans_ge2_min_cov_mult', 1.0))
    topk_w = float(config.get('answer_aux', {}).get('topk_listwise_weight', 0.0))
    aux_enabled = config.get('answer_aux', {}).get('enabled', False)

    for sample in tqdm(
        data_loader,
        desc='Train',
        leave=False,
        dynamic_ncols=True,
        mininterval=0.2,
        file=sys.stderr,
    ):
        sample = prepare_stage2_sample(device, sample)
        if sample['x_stage2'].numel() == 0:
            continue

        logits = model(sample['x_stage2'])
        bce = soft_bce_loss(logits, sample['soft_labels'])
        pair = pairwise_logistic_loss(logits, sample, config)
        ans = answer_touch_bce_loss(logits, sample, config)
        min_ans = min_answer_coverage_loss(logits, sample, config)
        topk_ls = topk_listwise_surrogate_loss(logits, sample, config)
        loss = bce + pairwise_weight * pair
        if aux_enabled:
            n_ans = int(sample.get('num_answers', 1))
            touch_mult = 1.0
            min_cov_mult = 1.0
            if bucket_enable:
                touch_mult = n_ans1_touch_mult if n_ans <= 1 else n_ans_ge2_touch_mult
                min_cov_mult = n_ans1_min_cov_mult if n_ans <= 1 else n_ans_ge2_min_cov_mult
            loss = loss + aux_w * touch_mult * ans
            if min_aw > 0:
                loss = loss + min_aw * min_cov_mult * min_ans
            if topk_w > 0:
                loss = loss + topk_w * topk_ls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics['loss'].append(loss.item())
        metrics['bce_loss'].append(bce.item())
        metrics['pairwise_loss'].append(pair.item())
        metrics['answer_aux_loss'].append(ans.item())
        metrics['min_answer_loss'].append(min_ans.item())
        metrics['topk_listwise_loss'].append(topk_ls.item())

    reduced = {}
    for key, values in metrics.items():
        reduced[key] = float(sum(values) / len(values)) if values else 0.0
    return reduced


def choose_target_metric(config, eval_metrics):
    ev = config.get('eval') or {}
    target_k = ev.get('target_k')
    ck = str(ev.get('checkpoint_metric', 'answer_recall')).lower()
    if target_k is None:
        target_k = max(config['eval']['k_list'])

    ar_key = f'answer_recall@{target_k}'
    fc_key = f'full_cov@{target_k}'
    ar = float(eval_metrics.get(ar_key, 0.0))
    fc = float(eval_metrics.get(fc_key, 0.0))

    if ck == 'full_cov':
        return fc
    if ck == 'combined':
        beta = float(ev.get('combined_beta', 0.5))
        return beta * ar + (1.0 - beta) * fc
    if ck == 'answer_recall' and ar_key in eval_metrics:
        return ar
    if ar_key in eval_metrics:
        return ar
    eval_k = max(config['eval']['k_list'])
    shortest_key = f'shortest_recall@{eval_k}'
    positive_key = f'positive_recall@{eval_k}'
    if shortest_key in eval_metrics:
        return eval_metrics[shortest_key]
    return eval_metrics.get(positive_key, 0.0)


def main(args):
    config = load_yaml(resolve_config_path(args.dataset, args.config))
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tr = config.get('train') or {}
    train_ds_kw = {}
    if tr.get('oversample_multi_answer', False):
        train_ds_kw['oversample_multi_answer'] = True
        train_ds_kw['multi_answer_extra_copies'] = int(tr.get('multi_answer_extra_copies', 1))

    train_set = Stage2CacheDataset(
        cache_dir=config['cache']['output_dir'],
        split='train',
        filter_all_negative=True,
        max_samples=args.max_train_samples,
        **train_ds_kw,
    )
    val_set = Stage2CacheDataset(
        cache_dir=config['cache']['output_dir'],
        split='val',
        filter_all_negative=False,
        max_samples=args.max_val_samples or config['eval']['max_eval_samples'] or None,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        collate_fn=collate_stage2_reranker,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_stage2_reranker,
    )

    if len(train_set) == 0:
        raise ValueError('Training cache is empty after filtering all-negative samples.')

    input_dim = build_stage2_inputs(train_set[0]).shape[-1]
    model = Stage2TripleReranker(
        input_dim=input_dim,
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
    ).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])

    run_name = args.run_name
    if run_name is None:
        ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
        run_name = f"{config['train']['save_prefix']}_{ts}"
    run_dir = os.path.join(config['train']['output_dir'], run_name)
    os.makedirs(run_dir, exist_ok=True)

    wandb_run = wandb.init(
        project='stage2_reranker',
        name=run_name,
        group=str(config.get('dataset', {}).get('name', 'unknown')),
        config=config,
        dir=os.path.join(run_dir, 'wandb'),
    )

    best_val_metric = float('-inf')
    num_patient_epochs = 0
    best_checkpoint_path = os.path.join(run_dir, 'cpt.pth')

    for epoch in tqdm(
        range(config['train']['num_epochs']),
        desc='Epochs',
        dynamic_ncols=True,
        mininterval=0.2,
        file=sys.stderr,
    ):
        train_metrics = train_epoch(config, device, train_loader, model, optimizer)
        val_metrics = eval_epoch(config, device, val_loader, model)
        target_metric = choose_target_metric(config, val_metrics)

        if target_metric > best_val_metric:
            best_val_metric = target_metric
            num_patient_epochs = 0
            checkpoint = {
                'config': config,
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'best_val_metric': best_val_metric,
                'epoch': epoch,
            }
            torch.save(checkpoint, best_checkpoint_path)
        else:
            num_patient_epochs += 1

        log_dict = {
            'epoch': epoch,
            'train/bce_loss': float(train_metrics.get('bce_loss', 0.0)),
            'train/pairwise_loss': float(train_metrics.get('pairwise_loss', 0.0)),
            'train/answer_aux_loss': float(train_metrics.get('answer_aux_loss', 0.0)),
            'train/min_answer_loss': float(train_metrics.get('min_answer_loss', 0.0)),
            'train/loss': float(train_metrics.get('loss', 0.0)),
            'val/best_val_metric': float(best_val_metric),
            'val/target_metric': float(target_metric),
            'val/num_patient_epochs': float(num_patient_epochs),
        }
        for k, v in val_metrics.items():
            log_dict[f'val/{k}'] = float(v)
        for k, v in train_metrics.items():
            if k in {'loss', 'bce_loss', 'pairwise_loss', 'answer_aux_loss', 'min_answer_loss'}:
                continue
            log_dict[f'train/{k}'] = float(v)

        wandb.log(log_dict)

        if num_patient_epochs >= config['train']['patience']:
            break

    wandb_run.finish()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config path override')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    args = parser.parse_args()

    main(args)
