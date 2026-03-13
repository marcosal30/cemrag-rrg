import argparse
import torch
import json
import os
import csv

from constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils import disable_torch_init
from mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

def image_parser(image_path,sep):
    out = image_path.split(sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(image_path, query, predictions_file,tokenizer, model, image_processor, context_len, all_preds, i):
    sep=','
    temperature=0
    conv_mode_args=None
    top_p=None
    num_beams=1
    max_new_tokens=1024

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        elif DEFAULT_IMAGE_TOKEN in qs:
            qs=qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
       

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv_mode = "mistral_instruct"
    
    if conv_mode_args is not None and conv_mode != conv_mode_args:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, conv_mode_args, conv_mode_args
            )
        )
    else:
        conv_mode_args = conv_mode

    conv = conv_templates[conv_mode_args].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(image_path,sep)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    )
    images_tensor=images_tensor[0].unsqueeze(0).to(model.device, dtype=torch.bfloat16)
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        
        model = model.to(torch.bfloat16)
        model=model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    with open(predictions_file, 'a') as file:
        file.write(outputs + "\n")

    all_preds[str(i)] = outputs

    return all_preds


if __name__ == "__main__":

    image_directory="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationData/mimic-cxr-jpg/"
    predictions_file="predictionsTestLLaVAMedFTMimicTot.txt"
    json_predictions_file = "predictionsTestLLaVAMedFTMimicTot.json"
    model_path="/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/bioRAG/LLaVAMed/checkpoints/llava-med-finetune_loraMimicTot"
    test_json='./data/mimic-cxr_test.json'
    model_base='/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/ReportGenerationModels/llava-med-v1.5-mistral-7b'
    

    # Model
    disable_torch_init()
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        print('esiste')
    else:
        print('non esiste')
        
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    all_preds = {}

    with open(test_json, 'r') as f:
        val_data = json.load(f)
    i=0
    for entry in val_data:
        image_id = entry.get('image', None)
        query=entry.get('conversations')[0]['value']
        image_path=image_directory+image_id
        all_preds=eval_model(image_path,query,predictions_file, tokenizer, model, image_processor, context_len, all_preds,i)
        i+=1

    with open(json_predictions_file, 'w', encoding='utf-8') as jf:
        json.dump(all_preds, jf, indent=2, ensure_ascii=False)

    print(f"Wrote {len(all_preds)} entries to {json_predictions_file}")