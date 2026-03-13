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
│   └── llava_med/             # LLaVA-Med + Mistral-7B
│
├── cemrag/                    # Core CEMRAG framework
│   ├── encoders/cxrclip/      # CXR-CLIP encoder (SwinTransformer + BioClinicalBERT)
│   ├── concepts/
│   │   ├── splice/            # SpLiCE sparse decomposition (ADMM solver)
│   │   └── vocab/             # Medical bigram/monogram vocabularies
│   └── retrieval/             # FAISS-based retrieval
│
├── scripts/                   # Pipeline steps (run in order)
│   ├── 1_compute_embeddings.py
│   ├── 2_compute_mean.py
│   ├── 3_run_splice.py
│   ├── 4_build_index.py
│   └── 5_train_and_eval.py
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

For clinical evaluation metrics, install separately:
- **CheXBert**: https://github.com/stanfordmlgroup/CheXBert
- **RadGraph**: https://github.com/jackboyla/radgraph

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

### 5. Train (SFT) or evaluate (Zero-Shot)
```bash
# Zero-Shot
python scripts/5_train_and_eval.py --config configs/mimic_cxr/cemrag_zeroshot.yaml

# Supervised Fine-Tuning (4x A100 GPUs)
torchrun --nproc_per_node=4 scripts/5_train_and_eval.py \
    --config configs/mimic_cxr/cemrag_sft.yaml
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
