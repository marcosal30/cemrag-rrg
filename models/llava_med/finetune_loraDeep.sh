#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 02-00:00:00
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out

# =============================================================================
# Configure these variables for your environment
# =============================================================================
VENV_DIR=/path/to/your/venv           # path to your Python virtual environment
PROJECT_DIR=/path/to/cemrag-rrg/models/llava_med
LLM_PATH=/path/to/llava-med-v1.5-mistral-7b
TRAIN_DATA=/path/to/data/mimic-cxr_train_cemrag.json
IMAGE_DIR=/path/to/mimic-cxr-jpg
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

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path "$LLM_PATH" \
    --version mistral_instruct \
    --data_path "$TRAIN_DATA" \
    --image_folder "$IMAGE_DIR" \
    --bf16 True \
    --output_dir ./checkpoints/llava-med-finetune_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to wandb

deactivate
