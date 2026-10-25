import random

import numpy as np
from torch.utils.data import get_worker_info


class WorkerRandom:
    def __init__(self, seed: int, process_index: int = 0) -> None:
        self.seed = seed
        self.process_index = process_index
        self._state: tuple[int, random.Random] | None = None

    def get(self) -> random.Random:
        # rebuilt in each dataloader worker, so forked workers never share one copied state
        worker = get_worker_info()
        worker_key = 0 if worker is None else worker.id + 1
        if self._state is None or self._state[0] != worker_key:
            entropy = [self.seed, self.process_index, worker_key]
            self._state = worker_key, random.Random(int(np.random.SeedSequence(entropy).generate_state(1)[0]))

        return self._state[1]
