from typing import Any, Dict, Optional, Tuple, Union

from multiprocessing import Pipe
from multiprocessing.queues import Queue
from multiprocessing.shared_memory import ShareableList, SharedMemory
from multiprocessing.synchronize import Lock

import numpy as np
import torch as th
from torch import nn


class SharedModel:
    def __init__(self, model: nn.Module, shared_mem: Optional[dict[str, SharedMemory]] = None) -> None:
        self.version = 0
        model.share_memory()
        th.Tensor.share_memory_

    def update(self, model: nn.Module) -> None:
        self.version += 1

    def get_model(self) -> nn.Module: ...
