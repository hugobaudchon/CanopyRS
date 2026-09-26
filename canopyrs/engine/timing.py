"""Per-stage timing for an inference loop."""

import time

import torch
from tqdm import tqdm


class InferTimer:
    """Where an inference loop spent its time, on one line.

    Iterate with ``batches``, ``mark`` after each step of the body (and after the loop, for tail
    work), then ``report``::

        timer = InferTimer("Inferring detector...")
        for object_ids, images in timer.batches(loader):
            images = [img.to(self.device) for img in images]
            timer.mark("prep")
            results.extend(self.forward(images))
            timer.mark("gpu")
        boxes, scores, classes = detector_result_to_lists(results)
        timer.mark("post")
        timer.report()

    Every key is main-process wall time and they sum to the whole loop: the marked stages, plus
    ``data_wait`` (blocked on the loader) and ``other`` (unmarked — should stay near zero, so a
    nonzero one means a missing ``mark``). Marks sync CUDA, so async kernels are billed to the stage
    that launched them.
    """

    def __init__(self, desc):
        self.desc = desc
        self.stages = {}                              # stage -> seconds, in the order first marked
        self.data_wait = 0.0
        self.other = 0.0
        self.n_batches = 0
        self.workers = 0
        self._sync = torch.cuda.is_available()
        self._last = time.perf_counter()

    def batches(self, loader):
        """``tqdm(loader)``, charging the block on each batch to ``data_wait``."""
        self.workers = loader.num_workers
        self._last = time.perf_counter()
        for batch in tqdm(loader, desc=self.desc, leave=True):
            self.data_wait += self._split()           # blocked until this batch arrived
            yield batch                               # the caller runs the model, marking stages
            self.other += self._split()               # body time past the caller's last mark
            self.n_batches += 1
        self.data_wait += self._split()

    def mark(self, stage):
        """Charge the time since the last mark (or since the batch arrived) to ``stage``."""
        if self._sync:
            torch.cuda.synchronize()
        self.stages[stage] = self.stages.get(stage, 0.0) + self._split()

    def report(self):
        if not self.n_batches:
            return
        self.other += self._split()                   # tail work the caller didn't mark
        wall = sum(self.stages.values()) + self.data_wait + self.other
        timings = {"data_wait": self.data_wait, **self.stages, "other": self.other}
        parts = [f"{stage} {s:.1f}s ({s / wall:.0%})" for stage, s in timings.items()]
        print(f"{self.desc.rstrip('. ')} timing: {self.n_batches} batches in {wall:.1f}s "
              f"({self.workers} loader workers) | " + " | ".join(parts))

    def _split(self):
        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now
        return elapsed
