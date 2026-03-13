import os
from .clip_encoder import CLIPVisionTower
from .open_clip_encoder import OpenCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower_config = getattr(vision_tower_cfg, 'mm_vision_tower_config', getattr(vision_tower_cfg, 'vision_tower_config', None))
    vision_tower_checkpoint = getattr(vision_tower_cfg, 'mm_vision_tower_checkpoint', getattr(vision_tower_cfg, 'vision_tower_checkpoint', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf-hub:") or vision_tower_config and vision_tower_checkpoint:
        return OpenCLIPVisionTower(
            vision_tower, args=vision_tower_cfg, vision_tower_config=vision_tower_config, vision_tower_checkpoint=vision_tower_checkpoint, **kwargs
        )

    raise ValueError(f'Unknown vision tower: {vision_tower}')
