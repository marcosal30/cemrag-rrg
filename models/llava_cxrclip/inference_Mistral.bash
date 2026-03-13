#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-577 -p alvis
#SBATCH -N 1 --gpus-per-node=A100:1
#SBATCH -t 00-05:00:00
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

python llava/train/inference_Mistral.py 

#Deactivate venv
deactivate