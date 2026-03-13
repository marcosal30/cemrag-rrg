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

cd /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG_venv || exit

source bin/activate

#Executes the code

cd /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/LLaVA-Rad || exit

export WANDB_MODE=offline

python llava/train/train_mem.py --model_name_or_path /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationModels/Vicuna-7b-v1.3 --version v1 --data_path /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/CXR-CLIP_LLaVA/data/mimic_trainKaggle.json --validation_data_path /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/CXR-CLIP_LLaVA/data/mimic_valKaggle.json --image_folder /mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/small_mimic_kaggle --vision_tower CXRCLIP --tune_mm_mlp_adapter True --mm_vision_select_layer -2 --mm_projector_type linear --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 False --output_dir ./checkpoints/llava-Mistral-7b-pretrainMimic --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --evaluation_strategy epoch --save_strategy epoch --save_steps 2400 --save_total_limit 1 --learning_rate 2e-3 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --tf32 False --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True --report_to wandb

#Deactivate venv
deactivate