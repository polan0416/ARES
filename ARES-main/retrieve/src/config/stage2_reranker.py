import os
import pydantic
import yaml
from typing import Optional

from .base import EnvYaml


class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str


class Stage1Yaml(pydantic.BaseModel):
    checkpoint_path: str
    retrieval_dir: str
    candidate_top_k: int


class LabelYaml(pydantic.BaseModel):
    dir: str
    direct_label: float = 1.0
    bridge_label: float = 0.7
    weak_label: float = 0.1
    silver_label: Optional[float] = None


class CacheYaml(pydantic.BaseModel):
    output_dir: str
    tensor_dtype: str
    distance_cap: int


class ModelYaml(pydantic.BaseModel):
    hidden_dims: str
    dropout: float


class OptimizerYaml(pydantic.BaseModel):
    lr: float
    weight_decay: float = 0.0


class EvalYaml(pydantic.BaseModel):
    k_list: str
    max_eval_samples: int = 0
    target_k: int = 100
    checkpoint_metric: str = 'combined'
    combined_beta: float = 0.5


class PairwiseYaml(pydantic.BaseModel):
    enabled: bool
    weight: float
    max_pos: int
    max_neg: int
    hard_negative_top_k: int
    hard_negative_top_percent: float
    high_conf_min_topic_distance: int
    mid_band_enabled: bool = True
    mid_band_start_idx: int = 49
    mid_band_end_idx: int = 149


class TrainYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    batch_size: int
    save_prefix: str
    output_dir: str
    oversample_multi_answer: bool = False
    multi_answer_extra_copies: int = 1


class InferenceYaml(pydantic.BaseModel):
    use_cache: bool
    output_dir: str
    output_prefix: str
    fusion_weight_stage1: float = 0.65
    fusion_weight_stage2: float = 0.35
    fusion_mode: str = 'linear'
    stage2_logit_scale: float = 1.0


class AnswerAuxYaml(pydantic.BaseModel):
    enabled: bool = False
    weight: float = 0.4
    pos_weight_cap: float = 50.0
    min_answer_weight: float = 0.0
    multi_answer_min_cov_boost: float = 0.0
    multi_answer_touch_boost: float = 0.0
    bucket_enable: bool = False
    n_ans1_touch_mult: float = 1.0
    n_ans_ge2_touch_mult: float = 1.0
    n_ans1_min_cov_mult: float = 1.0
    n_ans_ge2_min_cov_mult: float = 1.0
    topk_listwise_weight: float = 0.0
    topk_listwise_fullcov_weight: float = 0.5
    topk_listwise_target_k: int = 0
    topk_listwise_tau: float = 1.0


class Stage2RerankerYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    stage1: Stage1Yaml
    label: LabelYaml
    cache: CacheYaml
    model: ModelYaml
    optimizer: OptimizerYaml
    eval: EvalYaml
    pairwise: PairwiseYaml
    train: TrainYaml
    inference: InferenceYaml
    answer_aux: AnswerAuxYaml = AnswerAuxYaml()


def resolve_config_path(dataset: str, config_override: Optional[str] = None) -> str:
    if config_override:
        return config_override
    return f'configs/stage2_reranker/{dataset}.yaml'


def apply_stage1_run_path(
    config: dict,
    run_path: str,
    base_dir: Optional[str] = None,
) -> dict:
    checkpoint_path = resolve_run_checkpoint(run_path, base_dir=base_dir)
    config['stage1']['checkpoint_path'] = checkpoint_path
    config['stage1']['retrieval_dir'] = os.path.dirname(checkpoint_path)
    return config


def resolve_run_checkpoint(run_path: str, base_dir: Optional[str] = None) -> str:
    if base_dir and not os.path.isabs(run_path):
        run_path = os.path.join(base_dir, run_path)

    run_path = os.path.normpath(os.path.expanduser(run_path))

    if os.path.isfile(run_path):
        return run_path
    return os.path.join(run_path, 'cpt.pth')


def get_label_config(config: dict) -> dict:
    if 'label' in config:
        return config['label']
    if 'silver' in config:
        return config['silver']
    raise KeyError('config missing label section')


def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'stage2_reranker'

    if 'silver' in yaml_data and 'label' not in yaml_data:
        yaml_data['label'] = yaml_data.pop('silver')

    config = Stage2RerankerYaml(**yaml_data).model_dump()
    config['model']['hidden_dims'] = [
        int(size) for size in config['model']['hidden_dims'].split(',') if size]
    config['eval']['k_list'] = [
        int(k) for k in config['eval']['k_list'].split(',') if k]
    return config
