import logging
import os
import random
import sys

import numpy as np
import torch
from transformers import set_seed as hf_set_seed

log = logging.getLogger(__name__)


def set_seed_everywhere(seed: int) -> None:
    """Set seed for all random number generators (that I could find)."""
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


def reexec_with_hash_seed(seed: int) -> None:
    """Restarts this process with ``PYTHONHASHSEED`` set when it isn't already, since it only applies at startup.

    The process is replaced in place (same pid, same arguments and environment), so launchers that track it by pid,
    like ``accelerate launch``, are unaffected.
    """
    if os.environ.get("PYTHONHASHSEED") == str(seed):
        return

    log.info(f"restarting with PYTHONHASHSEED={seed}")
    os.execve(sys.executable, sys.orig_argv, {**os.environ, "PYTHONHASHSEED": str(seed)})
