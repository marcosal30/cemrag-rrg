import torch
from torch.utils.data import DataLoader
from splice import splice
import os
import json
import glob
import os
import torch
from omegaconf import DictConfig, OmegaConf
from cxrclip.data.datamodule import DataModule
from cxrclip.model import build_model
import sys
from torchvision import transforms
from PIL import Image
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd


class CLIPDataset(torch.utils.data.Dataset):
    """
    Custom Torch dataset for our CLIP model that loads instances from disk
    """

    def __init__(self, ids, data_folder):
        self.ids = ids
        self.data_folder = data_folder

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = torch.load(os.path.join(self.data_folder, f"{id}_image.pth")).squeeze()
        caption = torch.load(os.path.join(self.data_folder, f"{id}_caption.pth")).squeeze()
        return image, caption, id
    
def decompose_image(device,image_path,splicemodel,vocab):

    transform = transforms.Compose([
                transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(size=(224, 224))
            ])  
    #img_directory_path="/Users/marcosalme/Desktop/bioRAG/MIMIC-CXR/small_mimic_kaggle/"
    img_directory_path="/Users/marcosalme/Desktop/bioRAG/MIMIC-CXR/shorter_side/"
    path=img_directory_path+image_path
    img = Image.open(path).convert("RGB")
    img=transform(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(img)
    img=img.unsqueeze(0)

    weights, l0_norm, cosine = splice.decompose_image(img, splicemodel, device)

    top_concepts = 5  # imposta il numero di concetti da selezionare
    topk_weights, topk_indices = torch.topk(weights, k=top_concepts, dim=1)

    splice_decomposition = []

    for idx in topk_indices[0]:
        weight = weights[0, idx.item()].item()
        if weight == 0:
            continue
        splice_decomposition.append(vocab[idx.item()])

    return splice_decomposition



def decompose(device, decompose_df, splicemodel, vocab):

    data=[]
    for index, row in decompose_df.iterrows():
        splice_decs= decompose_image(device,row['path'],splicemodel,vocab)
        item = {
                "id": row['dicom_id'],
                "image": row['path'],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n Provide a description of the findings in the radiology image given these possible keywords: " + ", ".join(splice_decs)
                    },
                    {
                         "from": "gpt",
                         "value": row['text']
                    }
                        ]
                }
        
        data.append(item)
        
    return data

    
def main(cfg: DictConfig):

    image_mean = torch.load('./mean_embedding_mimicTot.pt')
    #image_mean = torch.load('./mean_embedding.pt')
    #vocab_path = '/Users/marcosalme/Desktop/bioRAG/SPLICE_code/combined_top_terms.txt' #IU-xray
    vocab_path = './vocab/filtered_bigramsMimicTot.txt' #MIMIC
    vocab_size=200
    device='cpu'
    print(cfg.test.checkpoint)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #OmegaConf.resolve(cfg)
    if type(cfg.test.checkpoint).__name__ == "ListConfig":
        ckpt_paths = cfg.test.checkpoint
    elif os.path.isdir(cfg.test.checkpoint):
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.test.checkpoint, "*.tar")))
    else:
        ckpt_paths = sorted(glob.glob(cfg.test.checkpoint))
    cfg_dict = OmegaConf.to_container(cfg)
    ckpt = torch.load(ckpt_paths[0], map_location="cpu",weights_only=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
            data_path="./data/mimic-cxr_train.csv"
        )

    # load model
    tokenizer=datamodule.tokenizer
    model = build_model(
        model_config=ckpt_config["model"], loss_config=ckpt_config["loss"], tokenizer=datamodule.tokenizer
    )

    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()

    decompose_df=pd.read_csv('./data/mimic-cxr_train.csv')

     ## Compute embedded dictionary, mean-center and normalize
    concepts = []
    vocab=[]
    with open(vocab_path, "r") as f:
        lines = f.readlines()[:vocab_size]
        for line in lines:
            tokens = tokenizer(line.strip())
            vocab.append(line.strip())
            with torch.no_grad():
                tokens = {key: torch.tensor(value).unsqueeze(0) for key, value in tokens.items()} #from list to tensor
                text_emb=model.encode_text(tokens)
                text_emb = model.text_projection(text_emb) if model.projection else text_emb
                text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
                concepts.append(text_emb)

    concepts = torch.nn.functional.normalize(torch.stack(concepts).squeeze(), dim=1)
    concepts = torch.nn.functional.normalize(concepts-torch.mean(concepts, dim=0), dim=1)

    splicemodel = splice.SPLICE(image_mean, concepts, clip=model, device=device,l1_penalty = 0.1)

    data=decompose(device,decompose_df,splicemodel,vocab)
    output_filename='mimic-cxr_train_spliceTerms.json'
    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing config file path argument.")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = OmegaConf.load(config_path)
    main(cfg)