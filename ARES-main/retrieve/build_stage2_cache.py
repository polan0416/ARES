#!/usr/bin/env python3
"""Build stage2 feature caches (requires stage1 checkpoint + retrieval_result + label JSON)."""

import argparse
import os
import sys
import time

import torch

# Allow running from retrieve/ without installing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.stage2_reranker import (
    apply_stage1_run_path,
    load_yaml,
    resolve_config_path,
)
from src.dataset.stage2_reranker import Stage2SampleBuilder


RETRIEVE_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, default=None,
                        help='Stage1 run directory or cpt.pth; overrides '
                             'stage1.checkpoint_path and stage1.retrieval_dir')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config path override')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_config_path(args.dataset, args.config))
    if args.path:
        apply_stage1_run_path(config, args.path, base_dir=RETRIEVE_ROOT)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache_root = config['cache']['output_dir']

    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    for split in splits:
        out_dir = os.path.join(cache_root, split)
        print(f'[{split}] Loading dataset, stage1 model, retrieval results...', flush=True)
        builder = Stage2SampleBuilder(config=config, split=split, device=device)
        num_samples = builder.count_samples(max_samples=args.max_samples)
        print(f'[{split}] Building cache for {num_samples} samples -> {out_dir}', flush=True)
        t0 = time.time()
        manifest = builder.build_cache(cache_dir=out_dir, max_samples=args.max_samples)
        elapsed = time.time() - t0
        num_samples = manifest.get('num_samples', 'unknown') if isinstance(manifest, dict) else 'unknown'
        print(
            f'[{split}] Finished -> {out_dir}  '
            f'(num_samples={num_samples}, elapsed={elapsed:.1f}s)',
            flush=True,
        )


if __name__ == '__main__':
    main()
