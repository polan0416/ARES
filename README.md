# ARES: Two-Stage Retrieval and LLM Reasoning for KGQA

## Pipeline

```text
Embedding
  -> Stage-1 MLP Retriever
  -> Stage-2 MLP Reranker
  -> LLM Retriever SFT
  -> LLM Reasoner SFT
  -> Retriever GRPO
```

## Environments

```bash
conda activate gte   # embedding
conda activate ret   # MLP retrievers
conda activate sft   # supervised fine-tuning      use llamafactory
conda activate rl    # GRPO training               use verl
conda activate rea    # reasoning

environment Pakage in ARES/ARES-main/environment
```

## 1. Stage-1 MLP Retriever

### 1.1 Compute Embeddings

```bash
conda activate gte
cd ARES/retrieve

python emb.py -d webqsp
```

### 1.2 Train

```bash
conda activate ret
cd ARES/retrieve

python train.py -d webqsp
# or
python train.py -d cwq
```

### 1.3 Inference

```bash
python inference.py \
  -p ARES/retrieve/webqsp_***/cpt.pth \
  --max_K 500 \
  --split all
```

CWQ example:

```bash
python inference.py \
  -p ARES/retrieve/cwq_***/cpt.pth \
  --max_K 500 \
  --split all
```

## 2. Stage-2 MLP Reranker

### 2.1 Build Cache

```bash
cd ARES/retrieve

python build_stage2_cache.py \
  -d webqsp \
  -p webqsp_*** \
  --split all
```

CWQ example:

```bash
python build_stage2_cache.py \
  -d cwq \
  -p cwq_*** \
  --split all
```

### 2.2 Train

```bash
python train_stage2.py -d webqsp
python train_stage2.py -d cwq
```

### 2.3 Inference

```bash
python inference_stage2.py \
  -d webqsp \
  -p stage2_results/stage2_webqsp_*** \
  --split all
```

## 3. LLM Retriever SFT

### 3.1 Build Training Data

```bash
conda activate sft
cd ARES/train/retriever/sft

python build_sft_data.py --datasets webqsp
```

### 3.2 Train

```bash
bash ARES/train/retriever/sft/train/sft_webqsp.sh
```

## 4. LLM Reasoner SFT

### 4.1 Build Top-100 Retrieval Data

```bash
conda activate sft

python ARES/train/retriever/sft/build_eval_data.py
```

### 4.2 Build KGQA Data

```bash
bash ARES/train/retriever/sft/eval/build_kgqa_webqsp.sh
bash ARES/train/retriever/sft/eval/build_kgqa_cwq.sh
```

### 4.3 Build Reasoner SFT Data

```bash
bash ARES/train/reasoner/build_kgqa_sft_data.sh
```

### 4.4 Train the Reasoner

```bash
bash ARES/train/reasoner/train/cwq/run_sft.sh
```

## 5. Retriever GRPO

### 5.1 Build GRPO Data

```bash
conda activate rl

python ARES/train/retriever/rl/build_grpo_data.py \
  --datasets webqsp
```

### 5.2 Start the Remote Reasoner

```bash
bash ARES/train/retriever/rl/launch_remote_answer_server_vllm.sh
```

### 5.3 Run GRPO Training

```bash
bash ARES/train/retriever/rl/run_webqsp_retriever_grpo.sh
```
## 6. Reasoning

```bash
conda activate rea
```

```bash
bash ARES/ARES-main/reason/eval_webqsp.sh
```