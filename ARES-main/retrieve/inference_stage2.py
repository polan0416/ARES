import os

import sys
import torch
from tqdm import tqdm

from src.config.stage2_reranker import (
    load_yaml,
    resolve_config_path,
    resolve_run_checkpoint,
)
from src.dataset.stage2_reranker import (
    Stage2CacheDataset,
    Stage2SampleBuilder,
    build_stage2_inputs,
    prepare_stage2_sample,
)
from src.model.stage2_reranker import Stage2TripleReranker, compute_inference_fusion_score
from src.setup import set_seed


RETRIEVE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _build_output_name(config, split):
    if split == 'test':
        return f"{config['inference']['output_prefix']}.pth"
    return f"{config['inference']['output_prefix']}_{split}.pth"


def _iterate_samples(config, split, use_cache, max_samples):
    if use_cache:
        return Stage2CacheDataset(
            cache_dir=config['cache']['output_dir'],
            split=split,
            filter_all_negative=False,
            max_samples=max_samples,
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    builder = Stage2SampleBuilder(config=config, split=split, device=device)
    return list(builder.iter_samples(max_samples=max_samples))


@torch.no_grad()
def rerank_split(
    config,
    checkpoint_path,
    split,
    output_dir,
    max_samples=None,
    use_cache=True,
    device_override=None,
    fusion_w1=None,
    fusion_w2=None,
):
    if device_override is not None:
        device = torch.device(device_override)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    samples = _iterate_samples(
        config=config,
        split=split,
        use_cache=use_cache,
        max_samples=max_samples,
    )
    if len(samples) == 0:
        raise ValueError(f'No samples for split={split}; check cache and retrieval_result alignment.')

    sample0 = samples[0]
    actual_dim = int(build_stage2_inputs(sample0).shape[-1])
    ckpt_dim = int(checkpoint.get('input_dim', -1))
    if ckpt_dim != actual_dim:
        raise ValueError(
            f'Checkpoint feature dim mismatch.\n'
            f'  --checkpoint file: {checkpoint_path}\n'
            f'  checkpoint input_dim={ckpt_dim}, current cache x_stage2 last dim={actual_dim}.\n'
            f'Common cause: using an old cpt trained before the de-anchor change '
            f'(often 4143); current code+cache should be 4140.\n'
            f'Point --checkpoint to cpt.pth from this retraining run, e.g.:\n'
            f'  stage2_runs/<your_run_name>/cpt.pth\n'
            f'Do not keep pointing at old dirs like stage2_ans_aux unless you still '
            f'use the old code and old cache.'
        )

    model = Stage2TripleReranker(
        input_dim=ckpt_dim,
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pred_dict = {}
    for sample in tqdm(
        samples,
        desc=f'Stage2 inference {split}',
        dynamic_ncols=True,
        mininterval=0.2,
        file=sys.stderr,
    ):
        prepared = prepare_stage2_sample(device, sample)
        logits = model(prepared['x_stage2'])
        stage2_probs = torch.sigmoid(logits)

        stage1_probs = prepared['f_stage1'][:, 0].to(dtype=torch.float32)

        score_final = compute_inference_fusion_score(
            prepared, logits, config, fusion_w1, fusion_w2,
        )
        sorted_indices = torch.argsort(score_final, descending=True)

        reranked = []
        for idx in sorted_indices.tolist():
            triple = sample['scored_triples'][idx]
            hop = triple[3] if len(triple) >= 4 else None
            reranked.append((
                triple[0],
                triple[1],
                triple[2],
                hop,
                float(score_final[idx].item()),
            ))

        pred_dict[sample['id']] = {
            'question': sample['question'],
            'scored_triples': reranked,
            'q_entity': sample['q_entity'],
            'q_entity_in_graph': sample['q_entity_in_graph'],
            'a_entity': sample['a_entity'],
            'a_entity_in_graph': sample['a_entity_in_graph'],
            'max_path_length': sample['max_path_length'],
            'target_relevant_triples': sample['target_relevant_triples'],
            'stage1_scores': [float(value) for value in stage1_probs[sorted_indices].tolist()],
            'stage2_scores': [float(value) for value in stage2_probs[sorted_indices].tolist()],
            'stage2_logits': [float(value) for value in logits[sorted_indices].tolist()],
            'final_scores': [float(value) for value in score_final[sorted_indices].tolist()],
            'candidate_triples_original_order': list(sample['scored_triples']),
        }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, _build_output_name(config, split))
    torch.save(pred_dict, out_path)
    print(f'Saved {len(pred_dict)} reranked samples to {out_path}')


def main(args):
    config = load_yaml(resolve_config_path(args.dataset, args.config))
    checkpoint_path = resolve_run_checkpoint(args.path, base_dir=RETRIEVE_ROOT)
    if args.fusion_w1 is not None:
        config['inference']['fusion_weight_stage1'] = float(args.fusion_w1)
    if args.fusion_w2 is not None:
        config['inference']['fusion_weight_stage2'] = float(args.fusion_w2)
    if args.fusion_mode is not None:
        config['inference']['fusion_mode'] = str(args.fusion_mode).strip()
    if args.stage2_logit_scale is not None:
        config['inference']['stage2_logit_scale'] = float(args.stage2_logit_scale)
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    if args.output_dir is None:
        output_dir = config['inference']['output_dir']
    else:
        output_dir = args.output_dir

    if args.use_cache is None:
        use_cache = config['inference']['use_cache']
    else:
        use_cache = args.use_cache

    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    for split in splits:
        rerank_split(
            config=config,
            checkpoint_path=checkpoint_path,
            split=split,
            output_dir=output_dir,
            max_samples=args.max_samples,
            use_cache=use_cache,
            device_override=args.device,
            fusion_w1=args.fusion_w1,
            fusion_w2=args.fusion_w2,
        )


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Stage2 run directory or cpt.pth, e.g. '
                             'stage2_runs/stage2_webqsp_Jun22-11:56:11')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config path override')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--use_cache', action='store_true', default=None)
    parser.add_argument('--online', dest='use_cache', action='store_false')
    parser.add_argument(
        '--fusion_w1',
        type=float,
        default=None,
        help='Override yaml fusion_weight_stage1 (use with --fusion_w2; for sweeps without editing config)',
    )
    parser.add_argument(
        '--fusion_w2',
        type=float,
        default=None,
        help='Override yaml fusion_weight_stage2',
    )
    parser.add_argument(
        '--fusion_mode',
        type=str,
        default=None,
        choices=['linear', 'logit_sum'],
        help='Override yaml: linear or logit_sum (latter uses sigmoid(l1+a*l2), ignores w1/w2)',
    )
    parser.add_argument(
        '--stage2_logit_scale',
        type=float,
        default=None,
        help='When fusion_mode=logit_sum, stage2 logit scale a in sigmoid(l1+a*l2); sweep on val',
    )
    args = parser.parse_args()

    main(args)
