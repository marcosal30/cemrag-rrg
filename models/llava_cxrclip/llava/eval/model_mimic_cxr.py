"""
A model worker executes the model.
"""
import os
import json
import math

import torch
import fire
from tqdm import tqdm
from PIL import Image, ImageFile
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import build_logger, disable_torch_init, data_loaders
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def create_batches(data, batch_size, group_by_length, tokenizer):
    if batch_size == 1 or not group_by_length:
        return [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    else:
        batches = []
        batch, batch_len = [], None
        for d in data:
            d_len = len(tokenizer(d["conversations"][0]['value']).input_ids)
            if batch_len is None or d_len == batch_len:
                batch_len = d_len
                batch.append(d)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch, batch_len = [], None
            else:
                assert len(batch)
                batches.append(batch)
                batch, batch_len = [d], d_len
        if len(batch):
            batches.append(batch)
        assert len(data) == sum(len(b) for b in batches)
        return batches


def eval_model(
        query_file: str,
        image_folder: str,
        conv_mode: str,
        prediction_file: str,
        model_path: str,
        model_base: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device: str = "cuda",
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        chunk_idx: int = 0,
        num_chunks: int = 1,
        batch_size: int = 8,
        loader: str = "default",
        group_by_length: bool = False,
    ):
    os.makedirs("logs", exist_ok=True)
    logger = build_logger("model_mimic_cxr", f"logs/model_mimic_cxr_{chunk_idx}.log")



    # load model
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    if not model_name.startswith("llavarad"):
        # "llava" needs to be in model_name to correctly load the model.
        raise ValueError(f"Model name {model_name} is not 'llavarad'.")
    logger.info(f"Loading the model {model_name} ...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device=device)

    # load data
    all_queries = data_loaders[loader](query_file)
    if group_by_length:
        all_queries = sorted(all_queries, key=lambda x: len(tokenizer(x["conversations"][0]['value']).input_ids))
    queries = get_chunk(all_queries, num_chunks, chunk_idx)
    logger.info(f"Loaded {len(queries)} / {len(all_queries)} ({chunk_idx}:{num_chunks}) examples.")

    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    pred_file = open(prediction_file, "w")
    log_prediction = True
    batches = create_batches(queries, batch_size, group_by_length, tokenizer)
    for batch_queries in tqdm(batches):
        batch_prompts = []
        batch_input_ids = []
        batch_images = []
        for query in batch_queries:
            q = query["conversations"][0]["value"]

            q = q.replace("<image>", "").strip()
            if model.config.mm_use_im_start_end:
                q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
            else:
                q = DEFAULT_IMAGE_TOKEN + '\n' + q

            conv= conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

            image = Image.open(os.path.join(image_folder, query["image"])).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            batch_prompts.append(prompt)
            batch_input_ids.append(input_ids)
            batch_images.append(image_tensor)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        

        with torch.inference_mode():
            batch_output_ids = model.generate(
                torch.stack(batch_input_ids).cuda(),
                images=torch.stack(batch_images).half().cuda(),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=256,
                use_cache=True).cpu()

            batch_outputs = tokenizer.batch_decode(
                batch_output_ids[:, len(batch_input_ids[0]):], skip_special_tokens=True
            )

        for query, prompt, outputs, input_ids, output_ids in zip(
            batch_queries, batch_prompts, batch_outputs, batch_input_ids, batch_output_ids):
            q = query["conversations"][0]["value"]
            ref = query["conversations"][1]["value"]
            input_token_len = input_ids.shape[0]
            n_diff_input_output = (input_ids != output_ids[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                logger.warning(f'{n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            if log_prediction:
                logger.info(f"id: {query['id']}")
                logger.info(f"query: {repr(q)}")
                logger.info(f"prompt: {repr(prompt)}")
                logger.info(f"reference: {repr(ref)}")
                logger.info(f"prediction: {repr(outputs)}")

            pred_file.write(json.dumps({"id": query["id"], "query": q, "reference": ref, "prediction": outputs}) + "\n")
            pred_file.flush()
        
        log_prediction = False

    pred_file.close()


if __name__ == "__main__":
    fire.Fire(eval_model)
