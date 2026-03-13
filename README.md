# CEMRAG: Concept-Enhanced Multimodal RAG for Radiology Report Generation

This is the **official repository** for the paper:

> **Concept-Enhanced Multimodal RAG: Towards Interpretable and Accurate Radiology Report Generation**

---

## Overview

CEMRAG is a framework for **interpretable and accurate radiology report generation** that challenges the assumed trade-off between transparency and performance in medical VLMs.

It integrates two complementary components into a unified pipeline:

- **Interpretable visual concept extraction** — visual embeddings are decomposed into sparse, clinically meaningful keywords using [SpLiCE](https://arxiv.org/abs/2402.09497) over a domain-specific medical vocabulary.
- **Multimodal Retrieval-Augmented Generation** — similar cases are retrieved from a vector database using [CXR-CLIP](https://arxiv.org/abs/2310.16527) embeddings and a [FAISS](https://github.com/facebookresearch/faiss) index.

Both components feed into a **hierarchical prompt** that directs the LLM to prioritize retrieved content aligned with observed visual features, improving both factual grounding and transparency.

![CEMRAG Framework](assets/methodsBioRag10.pdf)

---

## Repository Structure

```
cemrag-rrg/
├── models/                    # VLM backbone implementations
│   ├── llava_cxrclip/         # CXR-CLIP + Mistral-7B (LLaVA-style)
│   │   ├── pretrain.sh            # projection pretraining (single GPU)
│   │   ├── pretrainDeep.sh        # projection pretraining (multi-GPU, DeepSpeed)
│   │   ├── inference_llava-cxrclip.sh  # zero-shot inference
│   │   └── finetuning_loraDeep.sh      # SFT with LoRA (multi-GPU)
│   └── llava_med/             # LLaVA-Med + Mistral-7B
│       ├── inference_LLaVAMed.sh  # zero-shot inference
│       └── finetune_loraDeep.sh   # SFT with LoRA (multi-GPU)
│
├── cemrag/                    # Core CEMRAG framework
│   ├── encoders/
│   │   ├── cxrclip/           # CXR-CLIP encoder (SwinTransformer + BioClinicalBERT)
│   │   └── finetune_cxrclip.py  # LoRA fine-tuning entry point for IU X-Ray
│   ├── concepts/
│   │   ├── splice/            # SpLiCE sparse decomposition (ADMM solver)
│   │   └── vocab/             # Medical bigram/monogram vocabularies
│   └── retrieval/             # FAISS-based retrieval
│
├── scripts/                   # Pipeline steps (run in order)
│   ├── 1_compute_embeddings.py
│   ├── 2_compute_mean.py
│   ├── 3_run_splice.py        # → <split>_spliceTerms.json
│   ├── 4_build_index.py       # → <split>_rag.json
│   └── 5_hierarchical_prompt.py  # → <split>_cemrag.json  (merges 3 + 4)
│
├── configs/                   # YAML experiment configurations
│   ├── mimic_cxr/
│   └── iu_xray/
│
├── data/                      # Dataset instructions
│   └── README.md
│
└── assets/                    # Paper figures
```

---

## Installation

```bash
git clone https://github.com/marcosal30/cemrag-rrg.git
cd cemrag-rrg
pip install -r requirements.txt
```

---

## Pretrained Models

Download the following pretrained models before running experiments:

| Model | Source |
|---|---|
| **CXR-CLIP** (SwinTransformer + BioClinicalBERT) | [github.com/Soombit-ai/cxr-clip](https://github.com/Soombit-ai/cxr-clip) |
| **LLaVA-Med v1.5 Mistral-7B** | [huggingface.co/microsoft/llava-med-v1.5-mistral-7b](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) |
| **Mistral-7B-Instruct-v0.3** | [huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |

Set the model paths in the relevant shell scripts (see **Running Experiments** below) before launching.

---

## Data

See [`data/README.md`](data/README.md) for download and preprocessing instructions for **MIMIC-CXR** and **IU X-Ray**.

---

## Usage

Run the pipeline steps in order:

### 1. Compute image and text embeddings
```bash
python scripts/1_compute_embeddings.py
```

### 2. Compute mean embeddings (normalization)
```bash
python scripts/2_compute_mean.py
```

### 3. Run SpLiCE concept extraction
```bash
python scripts/3_run_splice.py
```

### 4. Build the FAISS retrieval index
```bash
python scripts/4_build_index.py
```

### 5. Build the CEMRAG hierarchical prompt

Merges the SpLiCE keywords (step 3) with the retrieved reports (step 4) into the hierarchical prompt format used by CEMRAG. Run once per split.

```bash
python scripts/5_hierarchical_prompt.py \
    --rag    data/mimic-cxr_train_rag.json \
    --splice data/mimic-cxr_train_spliceTerms.json \
    --output data/mimic-cxr_train_cemrag.json
```

The output JSON is the training / inference data for the CEMRAG prompting strategy. Pass it via `--data_path` in the experiment scripts.

### 7. Fine-tune CXR-CLIP with LoRA on IU X-Ray (optional)
```bash
# Single GPU
python -m cemrag.encoders.finetune_cxrclip configs/iu_xray/finetune_lora.yaml

# Multi-GPU (torchrun)
torchrun --nproc_per_node=4 -m cemrag.encoders.finetune_cxrclip \
    configs/iu_xray/finetune_lora.yaml
```

---

## Running Experiments

Edit the path variables at the top of each script before running. Scripts are formatted for SLURM (Alvis cluster) — remove or replace `module load` and `#SBATCH` directives for other environments.

### Zero-Shot setting

**LLaVA-Med** (inference only — uses the pretrained LLaVA-Med weights directly):
```bash
sbatch models/llava_med/inference_LLaVAMed.sh
```

**LLaVA-CXRClip** (requires pretraining the projection layer first):
```bash
# Step 1 — pretrain the projection layer
sbatch models/llava_cxrclip/pretrain.sh          # single GPU
sbatch models/llava_cxrclip/pretrainDeep.sh      # multi-GPU (DeepSpeed)

# Step 2 — run inference
sbatch models/llava_cxrclip/inference_llava-cxrclip.sh
```

### Supervised Fine-Tuning (SFT) setting

LLM fine-tuned with LoRA; projection layer fine-tuned; visual encoders frozen.

**LLaVA-CXRClip:**
```bash
sbatch models/llava_cxrclip/finetuning_loraDeep.sh
```

**LLaVA-Med:**
```bash
sbatch models/llava_med/finetune_loraDeep.sh
```

---

## Experimental Conditions

| Strategy | Visual | Concepts | Retrieved Reports | Interpretability |
|---|---|---|---|---|
| Image-Only | ✓ | — | — | ✗ |
| Concepts | ✓ | ✓ | — | Concept-level |
| RAG | ✓ | — | ✓ | Indirect |
| **CEMRAG** | ✓ | ✓ | ✓ | **Dual** |

Each strategy is evaluated under two training paradigms:
- **Zero-Shot**: all components frozen, prompt content only varies.
- **SFT**: LLM adapted via LoRA + projection layer fine-tuned; visual encoders frozen.

Benchmarks: **MIMIC-CXR** (in-domain retrieval) and **IU X-Ray** (cross-domain retrieval from MIMIC-CXR).

---

## Contact

**Marco Salmè** — marco.salme@unicampus.it
For questions, collaborations, or additional information about this work.
