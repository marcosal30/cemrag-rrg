"""
Trainer for LoRA fine-tuning of CXR-CLIP on IU X-Ray.

Applies LoRA adapters to selected Swin Transformer stages and BioClinicalBERT layers
while keeping all remaining parameters frozen. Supports single-GPU and DDP training.
"""
import logging
import math
import os
import shutil
from typing import Dict, Tuple, Any
from contextlib import nullcontext

import glob
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType

from . import util
from .data.datamodule import DataModule
from .loss import build_loss
from .model import build_model
from .scheduler import build_scheduler

log = logging.getLogger(__name__)


def unwrap_model(m):
    return m.module if isinstance(m, DDP) else m


def add_lora_to_cxrclip(
    model: nn.Module,
    *,
    vision_stages: Tuple[int, ...] = (3,),
    bert_last_layers: Tuple[int, ...] = (),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    unfreeze_layernorm: bool = True,
    unfreeze_projections: bool = False,
    verbose: bool = True,
):
    """
    Apply LoRA to a CXRClip model.

    Args:
        model: CXRClip instance.
        vision_stages: Swin stage indices to apply LoRA to (e.g. ``(3,)`` for last stage).
        bert_last_layers: BERT layer indices to apply LoRA to (e.g. ``(11,)``).
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: LoRA dropout probability.
        unfreeze_layernorm: If True, unfreeze all LayerNorm weights.
        unfreeze_projections: If True, unfreeze projection heads.
        verbose: Print trainable parameter summary.

    Returns:
        PEFT model with LoRA adapters and a patched forward compatible with CXRClip.
    """
    for p in model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["query", "key", "value", "dense"],
    )

    peft_model = get_peft_model(model, lora_cfg)
    base_model = peft_model.base_model.model

    def compatible_forward(batch=None, input_ids=None, attention_mask=None,
                           pixel_values=None, images=None, device=None, **kwargs):
        if batch is not None and isinstance(batch, dict):
            return base_model.forward(batch, device=device)

        if input_ids is not None or pixel_values is not None or images is not None:
            batch = {}
            if input_ids is not None:
                batch["text_tokens"] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask if attention_mask is not None
                    else torch.ones_like(input_ids),
                }
            if pixel_values is not None:
                batch["images"] = pixel_values
            elif images is not None:
                batch["images"] = images

            if "text_tokens" not in batch and "images" in batch:
                dev = batch["images"].device
                bs = batch["images"].shape[0]
                batch["text_tokens"] = {
                    "input_ids": torch.zeros(bs, 128, dtype=torch.long, device=dev),
                    "attention_mask": torch.ones(bs, 128, dtype=torch.long, device=dev),
                }
            elif "images" not in batch and "text_tokens" in batch:
                dev = batch["text_tokens"]["input_ids"].device
                bs = batch["text_tokens"]["input_ids"].shape[0]
                batch["images"] = torch.zeros(bs, 3, 224, 224, device=dev)

            return base_model.forward(batch, device=device)

        return base_model.forward(**kwargs)

    peft_model.forward = compatible_forward

    if unfreeze_layernorm:
        for n, p in peft_model.named_parameters():
            if any(x in n.lower() for x in ["layernorm", ".ln_", ".ln.", ".norm", "logit_scale"]):
                p.requires_grad = True

    if unfreeze_projections:
        for n, p in peft_model.named_parameters():
            if "projection" in n.lower() and "lora" not in n.lower():
                p.requires_grad = True

    swin_prefix = "base_model.model.image_encoder.image_encoder.encoder.layers."
    bert_prefix = "base_model.model.text_encoder.text_encoder.encoder.layer."

    for n, p in peft_model.named_parameters():
        if "lora_" in n:
            p.requires_grad = False

    for stage in vision_stages:
        stage_pattern = f"{swin_prefix}{stage}."
        for n, p in peft_model.named_parameters():
            if "lora_" in n and stage_pattern in n:
                p.requires_grad = True

    for layer in bert_last_layers:
        layer_pattern = f"{bert_prefix}{layer}."
        for n, p in peft_model.named_parameters():
            if "lora_" in n and layer_pattern in n:
                p.requires_grad = True

    if verbose:
        trainable = [(n, p.numel()) for n, p in peft_model.named_parameters() if p.requires_grad]
        n_trainable = sum(c for _, c in trainable)
        n_total = sum(p.numel() for p in peft_model.parameters())
        print(f"[LoRA] Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

        lora_examples = [n for n, _ in trainable if "lora_" in n][:10]
        if lora_examples:
            print("[LoRA] Active LoRA parameters (sample):")
            for name in lora_examples:
                print(f"  - {name}")

    return peft_model


class LoRACXRClip(nn.Module):
    """
    Wraps a CXRClip model with LoRA adapters while preserving the original interface.
    """

    def __init__(self, cxr_clip_model: nn.Module, lora_config: dict):
        super().__init__()
        self.model = add_lora_to_cxrclip(cxr_clip_model, **lora_config)

        base = self.model.base_model.model
        self.image_encoder = base.image_encoder
        self.text_encoder = base.text_encoder
        self.image_projection = base.image_projection
        self.text_projection = base.text_projection
        self.logit_scale = getattr(base, "logit_scale", torch.tensor(1.0))
        self.tokenizer = getattr(base, "tokenizer", None)

    def forward(self, batch, device=None):
        return self.model(batch, device=device)

    def encode_image(self, image):
        return self.model.base_model.model.encode_image(image)

    def encode_text(self, text_tokens):
        return self.model.base_model.model.encode_text(text_tokens)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        base = self.model.base_model.model
        self.logit_scale = getattr(base, "logit_scale", self.logit_scale.to(*args, **kwargs))
        return self

    def save_pretrained(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path: str, cxr_clip_model: nn.Module):
        from peft import PeftModel
        instance = cls.__new__(cls)
        super(LoRACXRClip, instance).__init__()
        instance.model = PeftModel.from_pretrained(cxr_clip_model, model_path)
        base = instance.model.base_model.model
        instance.image_encoder = base.image_encoder
        instance.text_encoder = base.text_encoder
        instance.image_projection = base.image_projection
        instance.text_projection = base.text_projection
        instance.logit_scale = getattr(base, "logit_scale", torch.tensor(1.0))
        instance.tokenizer = getattr(base, "tokenizer", None)
        return instance


def run(local_rank: int, cfg: Dict):
    distributed = local_rank != -1
    if distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"DistEnv: {util.GlobalEnv.get()}")

    # ── Data ─────────────────────────────────────────────────────────────────
    log.info(f"{device}: Load datasets")
    data_config = {}
    for split in ("data_train", "data_valid", "data_test"):
        if split in cfg:
            data_config[split.replace("data_", "")] = cfg[split]

    if "checkpoint" in cfg:
        ckpt_paths = sorted(glob.glob(os.path.join(cfg["checkpoint"], "*.tar")))
        ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
        ckpt_config = ckpt["config"]
        cfg["tokenizer"] = ckpt_config["tokenizer"]
        cfg["tokenizer"]["pretrained_model_name_or_path"] = "./tokenizer"
        cfg["transform"] = ckpt_config["transform"]

    datamodule = DataModule(
        data_config=data_config,
        dataloader_config=cfg["dataloader"],
        tokenizer_config=cfg.get("tokenizer"),
        loss_config=cfg["loss"],
        transform_config=cfg["transform"],
    )
    train_dataloader, train_sampler = datamodule.train_dataloader(distributed=distributed)
    valid_dataloaders = datamodule.valid_dataloader(distributed=distributed)

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"{device}: Build model")
    if "checkpoint" in cfg:
        cfg["model"] = ckpt_config["model"]
        model = build_model(cfg["model"], cfg["loss"], datamodule.tokenizer)
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model = build_model(cfg["model"], cfg["loss"], datamodule.tokenizer)
    model = model.to(device)

    if util.GlobalEnv.get().master:
        log.info(f"{device}: Model info:\n{model}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    log.info(f"{device}: Build loss, optimizer, scheduler")
    loss_func = build_loss(cfg["loss"])

    lora_config = {
        "vision_stages": (3,),
        "bert_last_layers": (11,),
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "unfreeze_layernorm": True,
        "unfreeze_projections": False,
        "verbose": True,
    }
    model = LoRACXRClip(model, lora_config)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3,
        weight_decay=0.01,
    )
    sched_cfg = cfg["scheduler"]["config"]
    if "total_epochs" in sched_cfg:
        sched_cfg["total_steps"] = len(train_dataloader) * sched_cfg["total_epochs"]
    if "warmup_epochs" in sched_cfg:
        warmup = sched_cfg["warmup_epochs"]
        sched_cfg["warmup_steps"] = (
            len(train_dataloader) * warmup if isinstance(warmup, int) else warmup
        )
    scheduler = build_scheduler(optimizer, cfg["scheduler"])

    if local_rank < 1:
        import nltk
        nltk.download("punkt")

    # ── Epochs ────────────────────────────────────────────────────────────────
    total_epochs = math.ceil(sched_cfg["total_steps"] / len(train_dataloader))

    util.GlobalEnv.get().summary_writer.train = util.DistSummaryWriter(
        cfg["base"]["output"]["tensorboard"] + "/train"
    )
    util.GlobalEnv.get().summary_writer.valid = util.DistSummaryWriter(
        cfg["base"]["output"]["tensorboard"] + "/valid"
    )
    util.GlobalEnv.get().summary_writer.global_step = 0
    util.GlobalEnv.get().summary_writer.train.add_text(
        "hyperparams/config",
        "\n".join(["\t" + line for line in OmegaConf.to_yaml(cfg).splitlines()]),
        0,
    )

    if util.GlobalEnv.get().master:
        os.makedirs(cfg["base"]["output"]["checkpoint"], exist_ok=True)

    if distributed:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    log.info(f"{device}: Start training ({total_epochs} epochs)")
    for epoch in range(total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss_dict = _train_epoch(
            model, device, loss_func, optimizer, scheduler,
            train_dataloader, epoch, total_epochs,
            sched_cfg["total_steps"],
        )
        log.info(f"Train loss epoch {epoch}: {train_loss_dict}")

        val_loss_dict = _validate_epoch(
            model, device, loss_func, valid_dataloaders,
            epoch, total_epochs, local_rank,
        )
        log.info(f"Val loss epoch {epoch}: {val_loss_dict}")

        # TensorBoard logging
        for k, v in train_loss_dict.items():
            util.GlobalEnv.get().summary_writer.train.add_scalar(
                f"loss_per_epoch/{k}", v, epoch + 1
            )

        avg_val_loss = {k.name: 0.0 for k in loss_func.loss_list}
        avg_val_loss["total"] = 0.0
        for data_name, loss_dict in val_loss_dict.items():
            for k, v in loss_dict.items():
                util.GlobalEnv.get().summary_writer.valid.add_scalar(
                    f"loss_per_epoch/{k}/{data_name}", v, epoch + 1
                )
                avg_val_loss[k] += v
        for k in avg_val_loss:
            avg_val_loss[k] /= len(valid_dataloaders)
            util.GlobalEnv.get().summary_writer.valid.add_scalar(
                f"loss_per_epoch/{k}", avg_val_loss[k], epoch + 1
            )

        # Checkpoint
        if util.GlobalEnv.get().master:
            filename = os.path.join(cfg["base"]["output"]["checkpoint"], "model")
            checkpoint = f"{filename}-last.tar"
            model_state = (
                model.state_dict() if local_rank == -1 else model.module.state_dict()
            )
            torch.save(
                {
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": cfg,
                    "epoch": epoch + 1,
                    "train_loss": train_loss_dict["total"],
                },
                checkpoint,
            )
            log.info(f"Epoch {epoch}: checkpoint saved")

    util.GlobalEnv.get().summary_writer.train.close()
    util.GlobalEnv.get().summary_writer.valid.close()
    log.info(f"{device}: Training complete")


def _train_epoch(
    model, device, loss_func, optimizer, scheduler,
    dataloader, epoch, total_epochs, total_step,
    print_step: int = 30, accum_steps: int = 8, grad_clip: bool = True,
):
    model.train()
    progress_iter = (
        tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch train]",
             total=len(dataloader))
        if util.GlobalEnv.get().local_rank < 1
        else enumerate(dataloader)
    )

    avg_loss = {"total": 0.0}
    for k in loss_func.loss_list:
        avg_loss[k.name] = 0.0

    optimizer.zero_grad(set_to_none=True)
    opt_step = 0

    for idx, batch in progress_iter:
        sync_ctx = (
            model.no_sync()
            if (hasattr(model, "no_sync") and (idx % accum_steps) != accum_steps - 1)
            else nullcontext()
        )
        with sync_ctx:
            outputs = model(batch, device)
            loss_dict = loss_func(**outputs, is_train=True)
            (loss_dict["total"] / accum_steps).backward()

        for k in loss_dict:
            avg_loss[k] += loss_dict[k].item()

        do_step = (idx % accum_steps == accum_steps - 1) or (idx == len(dataloader) - 1)
        if do_step:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            opt_step += 1

            global_step = getattr(scheduler, "_step_count", opt_step)
            util.GlobalEnv.get().summary_writer.global_step = global_step

            if idx % print_step == 0 and util.GlobalEnv.get().local_rank < 1:
                lr_now = [pg["lr"] for pg in optimizer.param_groups]
                for i, lr in enumerate(lr_now):
                    util.GlobalEnv.get().summary_writer.train.add_scalar(
                        f"hyperparam/lr-{i}", lr, global_step
                    )
                util.GlobalEnv.get().summary_writer.train.add_scalar(
                    "loss", loss_dict["total"], global_step
                )
                for k in loss_dict:
                    util.GlobalEnv.get().summary_writer.train.add_scalar(
                        f"loss/{k}", loss_dict[k], global_step
                    )
                progress_iter.set_postfix({
                    "lr": [f"{v:.8f}" for v in lr_now],
                    "loss": f"{loss_dict['total']:.6f}",
                })

            if total_step == global_step:
                break

    for k in avg_loss:
        avg_loss[k] /= len(dataloader)
    return avg_loss


def _validate_epoch(model, device, loss_func, dataloader_dict, epoch, total_epochs, local_rank,
                    print_step: int = 10):
    model.eval()
    loss_dict_per_dataset = {}
    with torch.no_grad():
        for data_name, dataloader in dataloader_dict.items():
            avg_loss = {"total": 0.0}
            for k in loss_func.loss_list:
                avg_loss[k.name] = 0.0

            progress_iter = (
                tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch valid]",
                     total=len(dataloader))
                if util.GlobalEnv.get().local_rank < 1
                else enumerate(dataloader)
            )

            for idx, batch in progress_iter:
                outputs = model(batch, device)
                loss_dict = loss_func(**outputs, is_train=False)

                if util.GlobalEnv.get().world_size > 1:
                    for k in loss_dict:
                        dist.all_reduce(loss_dict[k], dist.ReduceOp.SUM)
                        loss_dict[k] /= util.GlobalEnv.get().world_size

                for k in loss_dict:
                    avg_loss[k] += loss_dict[k].item()

                if (idx % print_step == 0 or idx == len(dataloader) - 1) and local_rank < 1:
                    progress_iter.set_postfix({"loss": f'{loss_dict["total"]:.6f}'})

            for k in avg_loss:
                avg_loss[k] /= len(dataloader)
            loss_dict_per_dataset[data_name] = avg_loss

    return loss_dict_per_dataset
