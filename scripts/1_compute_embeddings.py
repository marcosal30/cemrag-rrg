import glob
import os
import torch
from omegaconf import DictConfig, OmegaConf
from util.utils import seed_everything
from cxrclip.model import build_model
from cxrclip.data.datamodule import DataModule
import numpy as np
from tqdm import tqdm
import sys


def encode_image(model,device, image):
        with torch.no_grad():
            img_emb = model.encode_image(image.to(device))
            img_emb = model.image_projection(img_emb) if model.projection else img_emb
            img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
        #return img_emb.detach().cpu().numpy()
        return img_emb.detach().cpu()

def encode_text(model,device,ckpt_config,datamodule, text_token):
        if isinstance(text_token, str) or isinstance(text_token, list):
            text_token = datamodule.tokenizer(
                text_token, padding="longest", truncation=True, return_tensors="pt", max_length=ckpt_config["base"]["text_max_length"]
            )

        with torch.no_grad():
            text_emb = model.encode_text(text_token.to(device))
            text_emb = model.text_projection(text_emb) if model.projection else text_emb
            text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
        #return text_emb.detach().cpu().numpy()
        return text_emb.detach().cpu()

    
def main(cfg: DictConfig):
    print(cfg.test.checkpoint)
    seed_everything(cfg.test.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #OmegaConf.resolve(cfg)
    if type(cfg.test.checkpoint).__name__ == "ListConfig":
        ckpt_paths = cfg.test.checkpoint
    elif os.path.isdir(cfg.test.checkpoint):
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.test.checkpoint, "*.tar")))
    else:
        ckpt_paths = sorted(glob.glob(cfg.test.checkpoint))
    cfg_dict = OmegaConf.to_container(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = torch.load(ckpt_paths[0], map_location=device.type, weights_only=False)
    ckpt_config = ckpt["config"]
    ckpt_config["tokenizer"]['pretrained_model_name_or_path']='./tokenizer'
    
    data_config = {"test": {cfg_dict["defaults"][1]["data_test"]: {}}}  

    data_config["test"][cfg_dict["defaults"][1]["data_test"]]["normalize"] = "huggingface"
        
    ckpt_config["tokenizer"]['pretrained_model_name_or_path']='./tokenizer'
    datamodule = DataModule(
            data_config=data_config,
            dataloader_config=cfg_dict["defaults"][2]["dataloader"],
            tokenizer_config=ckpt_config["tokenizer"],
            transform_config=cfg_dict["transform"] if "transform" in cfg_dict else ckpt_config["transform"],
            data_path="./data/mimic_valKaggle.csv"
        )

        # load model
    model = build_model(
        model_config=ckpt_config["model"], loss_config=ckpt_config["loss"], tokenizer=datamodule.tokenizer
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()
     
    test_dataloader_dict = datamodule.test_dataloader()

    dataloader = test_dataloader_dict['mimic']

    image_embeddings = []
    text_embeddings = []
    texts = []
    out_path="embeddings/SmallMimic_val"
    for batch in tqdm(dataloader):
        image_id = batch['image_path'][0].split('/')[-1][:-4]
        img_emb = encode_image(model, device, batch["images"])
        torch.save(img_emb, os.path.join(out_path, image_id + "_image.pth"))
        image_embeddings.append(img_emb)
        
        if "texts" in batch:
            texts += batch["texts"]
        if "text_tokens" in batch:
            text_emb = encode_text(model,device,ckpt_config,datamodule,batch["text_tokens"])
            torch.save(text_emb, os.path.join(out_path, image_id + "_caption.pth"))
            text_embeddings.append(text_emb)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing config file path argument.")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = OmegaConf.load(config_path)
    main(cfg)
