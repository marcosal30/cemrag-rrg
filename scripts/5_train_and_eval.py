import logging
import os
import sys
import torch
from omegaconf import DictConfig, OmegaConf
from cxrclip.trainer import  run
from util.utils import seed_everything,convert_dictconfig_to_dict

log = logging.getLogger(__name__)

def main(cfg: DictConfig):

    OmegaConf.resolve(cfg)

    if "LOCAL_RANK" in os.environ:
        # for ddp
        # passed by torchrun or torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # for debugging
        local_rank = -1

    if local_rank < 1:
        log.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.base.seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    cfg = convert_dictconfig_to_dict(cfg)
    run(local_rank, cfg)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing config file path argument.")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = OmegaConf.load(config_path)
    main(cfg)