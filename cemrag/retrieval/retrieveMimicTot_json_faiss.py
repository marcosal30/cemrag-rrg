import torch
import faiss
from torch.utils.data import DataLoader
import os
import json
import glob
from omegaconf import DictConfig, OmegaConf
from cxrclip.data.datamodule import DataModule
from cxrclip.model import build_model
import sys
from torchvision import transforms
from PIL import Image
import pandas as pd

def build_faiss_index(embeddings: torch.Tensor, dim: int, use_gpu: bool = False):
    """
    Build a FAISS index for inner product search (cosine similarity).
    Assumes embeddings are already normalized.
    """
    emb_np = embeddings.cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(dim)
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(emb_np)
    return index


def encode_image(model, device, image_path, img_dir):
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    img = Image.open(os.path.join(img_dir, image_path)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = model.image_projection(emb) if model.projection else emb
        emb = emb / emb.norm(dim=1, keepdim=True)
    return emb.cpu()


def retrieve_with_faiss(query_df, retrieval_df, img_index, text_index, id_list, top_k):
    """
    Perform retrieval via FAISS for each query image embedding.
    id_list: list of ids in same order as built indices
    Returns structured RAG JSON entries.
    """
    data = []
    for _, row in query_df.iterrows():
        qid = row['id']
        qpath = row['path']
        q_emb = row['embedding'].unsqueeze(0).numpy().astype('float32')
        D_text, I_text = text_index.search(q_emb, top_k + 1)
        D_img, I_img = img_index.search(q_emb, top_k + 1)
        txt_ids = [id_list[i] for i in I_text[0] if id_list[i] != qid][:top_k]
        img_ids = [id_list[i] for i in I_img[0] if id_list[i] != qid][:top_k]
        texts_img = [f"{i+1}) " + retrieval_df.loc[retrieval_df['id']==iid, 'text'].iloc[0] for i, iid in enumerate(img_ids)]
        item = {
            "id": qid,
            "image": qpath[1:],
            "conversations": [
                {"from": "human", "value": "<image>\n Provide a description of the findings given these similar image findings: " + ", ".join(texts_img)},
                {"from": "gpt", "value": row['text']}
            ]
        }
        data.append(item)
    return data


def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = build_model(
        model_config=ckpt_config["model"], loss_config=ckpt_config["loss"], tokenizer=datamodule.tokenizer
    )

    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device)
    model.eval()
    # --- Explicitly separate query and retrieval dataframes ---
    train_df = pd.read_csv('./data/mimic-cxr_train.csv')
    val_df   = pd.read_csv('./data/mimic-cxr_train.csv')

    # Prepare IDs 
    train_df['id'] = train_df['path'].astype(str).apply(lambda x: x.split('/')[-1][:-4])
    val_df['id']   = val_df['path'].astype(str).apply(lambda x: x.split('/')[-1][:-4])


    # Load and normalize embeddings from training set
    data_folder = './embeddings/MimicTot_train'
    img_embs = []
    txt_embs = []
    for id_ in train_df['id']:
        img = torch.load(os.path.join(data_folder, f"{id_}_image.pth")).squeeze()
        txt = torch.load(os.path.join(data_folder, f"{id_}_caption.pth")).squeeze()
        img_embs.append(img / img.norm())
        txt_embs.append(txt / txt.norm())
    img_embs = torch.stack(img_embs)
    txt_embs = torch.stack(txt_embs)

    # Build FAISS indices on training embeddings
    dim = img_embs.shape[1]
    img_index  = build_faiss_index(img_embs, dim, use_gpu=torch.cuda.is_available())
    text_index = build_faiss_index(txt_embs, dim, use_gpu=torch.cuda.is_available())

    # Attach image embedding to val_df for queries
    val_data_folder = './embeddings/MimicTot_train'
    val_embeddings = []
    for id_ in val_df['id']:
        emb = torch.load(os.path.join(val_data_folder, f"{id_}_image.pth")).squeeze()
        emb = emb / emb.norm()
        val_embeddings.append(emb)
    val_df['embedding'] = val_embeddings

    # Run FAISS-based retrieval: use val_df as queries, train_df as retrieval corpus
    top_k = 4
    rag_data = retrieve_with_faiss(
        query_df=val_df,
        retrieval_df=train_df,
        img_index=img_index,
        text_index=text_index,
        id_list=train_df['id'].tolist(),
        top_k=top_k
    )

    # Save output
    with open('mimic-cxr_train_rag.json', 'w') as f:
        json.dump(rag_data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing config file path argument.")
        sys.exit(1)
    cfg = OmegaConf.load(sys.argv[1])
    main(cfg)
