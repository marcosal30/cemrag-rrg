#!/usr/bin/env bash
#SBATCH -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 02-00:00:00
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out

# =============================================================================
# Configure these variables for your environment
# =============================================================================
VENV_DIR=/path/to/your/venv                    # path to your Python virtual environment
PROJECT_DIR=/path/to/cemrag-rrg/models/llava_cxrclip
LLM_PATH=/path/to/Mistral-7B-Instruct-v0.3
TRAIN_DATA=/path/to/data/mimic-cxr_train_prompt2.json
IMAGE_DIR=/path/to/mimic-cxr-jpg
PRETRAIN_PROJECTOR=./checkpoints/llava-Mistral-7b-pretrain/mm_projector.bin
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
    --vision_tower CXRCLIP \
    --pretrain_mm_mlp_adapter "$PRETRAIN_PROJECTOR" \
    --mm_vision_select_layer -2 \
    --mm_projector_type linear \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-Mistral-7b-finetune_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
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
