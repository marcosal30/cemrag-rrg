#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 00-05:00:00
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out

# =============================================================================
# Configure these variables for your environment
# =============================================================================
VENV_DIR=/path/to/your/venv           # path to your Python virtual environment
PROJECT_DIR=/path/to/cemrag-rrg/models/llava_cxrclip
# =============================================================================

# Load modules — specific to the Alvis HPC cluster.
# Replace or remove if using a different environment.
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a
module load tokenizers/0.15.2-GCCcore-12.3.0
module load SentencePiece/0.2.0-GCC-12.3.0

source "$VENV_DIR/bin/activate"

cd "$PROJECT_DIR" || exit

export WANDB_MODE=offline

python llava/train/inference_llava-rad.py

deactivate
