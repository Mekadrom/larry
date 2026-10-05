import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import set_seed as hf_set_seed

log = logging.getLogger(__name__)


def load_yaml_config(yaml_file: Path) -> dict[str, Any]:
    with open(yaml_file) as stream:
        try:
            return yaml.safe_load(stream)
        except:
            log.exception(f"Error loading config file {yaml_file}:")
            raise


def set_seed_everywhere(seed: int):
    """Set seed for all random number generators."""
    random.seed(seed)

    np.random.seed(seed)

    # torch (cpu and cuda)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # hf transformers
    hf_set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
