#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4
#SBATCH -t 02-00:00:00
# Output files
#SBATCH --error=job_%J.err
#SBATCH --output=out_%J.out
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marco.salme@unicampus.it


# Load modules

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
module load Transformers/4.39.3-gfbf-2023a
module load tokenizers/0.15.2-GCCcore-12.3.0
module load SentencePiece/0.2.0-GCC-12.3.0
#module load einops/0.7.0-GCCcore-12.3.0



#activate venv

cd /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/LLaVA-RadMistral/llava-radMistral_venv || exit

source bin/activate

#Executes the code

cd /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/LLaVA-RadMistral || exit

export WANDB_MODE=offline

deepspeed llava/train/train_mem.py --deepspeed ./scripts/zero2.json --model_name_or_path /mimer/NOBACKUP/groups/snic2022-5-277/msalme/ReportGenerationFT/Mistral-7B-Instruct-v0.3 --version mistral_instruct --data_path /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/CXR-CLIP_LLaVA/data/mimic-cxr_train.json --image_folder /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationData/mimic-cxr-jpg --vision_tower biomedclip_cxr_518 --vision_tower_config llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json --vision_tower_checkpoint biomedclipcxr_518_checkpoint.pt --mm_projector_type mlp2x_gelu --tune_mm_mlp_adapter True --mm_vision_select_layer -2 --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir ./checkpoints/llava-Mistral-7b-pretrainMimicTot --num_train_epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --evaluation_strategy no --save_strategy epoch --save_steps 2400 --save_total_limit 1 --learning_rate 1e-3 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --report_to wandb