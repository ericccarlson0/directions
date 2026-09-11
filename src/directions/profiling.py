"""Run instrumentation: named wall-clock sections with forward-pass counters and memory.

Every pipeline stage runs inside ``Profiler.section(name, *scope)``; the key is
``name`` joined with the scope by colons (``"controls_profiles:antonym"``). A
section records wall seconds, process CPU seconds, the model backend's forward
and gradient counters (examples and batches) over its span and, on CUDA, the
peak allocated memory while it was open (reset at each outermost section).
``timings()`` is the flat ``{key: seconds}`` table written to
``metadata.json/timings_seconds``; ``report()`` is the structured version
written next to it as ``metadata.json/profile``.
"""

from __future__ import annotations

import os
import resource
import time
from contextlib import contextmanager
from typing import Any, Iterator

COUNTERS = ("forward_examples", "forward_batches", "gradient_examples", "gradient_batches")


class Profiler:
    def __init__(self, backend: Any = None) -> None:
        # anything with a ``counters`` dict over COUNTERS and a torch ``device`` (ModelBackend); may be set later
        self.backend = backend
        self.sections: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []
        self._depth = 0

    @property
    def _cuda(self) -> Any:
        dev = getattr(self.backend, "device", None)
        if dev is None or getattr(dev, "type", None) != "cuda":
            return None
        return dev

    def _counters(self) -> dict[str, int]:
        c = getattr(self.backend, "counters", None) or {}
        return {k: int(c.get(k, 0)) for k in COUNTERS}

    @staticmethod
    def _cgroup_throttled_seconds() -> float | None:
        """CPU time this container was throttled by its cgroup quota (cgroup v2 ``cpu.stat``), if readable."""
        try:
            for line in open("/sys/fs/cgroup/cpu.stat"):
                if line.startswith("throttled_usec"):
                    return int(line.split()[1]) / 1e6
        except OSError:
            return None
        return None

    @staticmethod
    def _host_state() -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            out["load_average_1m"] = round(os.getloadavg()[0], 2)
        except (OSError, AttributeError):
            pass
        try:
            out["cgroup_cpu_max"] = open("/sys/fs/cgroup/cpu.max").read().strip()
        except OSError:
            pass
        try:
            out["sched_cpus"] = len(os.sched_getaffinity(0))
        except (OSError, AttributeError):
            pass
        return out

    def _peak_gpu_mb(self) -> float | None:
        if self._cuda is None:
            return None
        import torch

        return float(torch.cuda.max_memory_allocated(self._cuda)) / 2**20

    @contextmanager
    def section(self, name: str, *scope: Any) -> Iterator[None]:
        key = ":".join((name, *(str(s) for s in scope)))
        if self._depth <= 1 and self._cuda is not None:  # a stage under the run-level section: its own peak
            import torch

            torch.cuda.reset_peak_memory_stats(self._cuda)
        before = self._counters()
        thr0 = self._cgroup_throttled_seconds()
        t0, c0 = time.perf_counter(), time.process_time()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            thr1 = self._cgroup_throttled_seconds()
            after = self._counters()
            rec = self.sections.get(key)
            if rec is None:
                rec = self.sections[key] = {"seconds": 0.0, "cpu_seconds": 0.0, "calls": 0, **{k: 0 for k in COUNTERS}}
                self.order.append(key)
            rec["seconds"] += wall
            rec["cpu_seconds"] += cpu
            rec["calls"] += 1
            if thr0 is not None and thr1 is not None:
                rec["throttled_seconds"] = rec.get("throttled_seconds", 0.0) + (thr1 - thr0)
            for k in COUNTERS:
                rec[k] += after[k] - before[k]
            peak = self._peak_gpu_mb()
            if peak is not None:
                rec["peak_gpu_mb"] = max(rec.get("peak_gpu_mb", 0.0), peak)

    def timings(self) -> dict[str, float]:
        """Flat ``{section: wall seconds}`` in first-entry order."""
        return {k: round(self.sections[k]["seconds"], 3) for k in self.order}

    def report(self) -> dict[str, Any]:
        sections = {}
        for k in self.order:
            rec = dict(self.sections[k])
            rec["seconds"] = round(rec["seconds"], 3)
            rec["cpu_seconds"] = round(rec["cpu_seconds"], 3)
            if "peak_gpu_mb" in rec:
                rec["peak_gpu_mb"] = round(rec["peak_gpu_mb"], 1)
            if "throttled_seconds" in rec:
                rec["throttled_seconds"] = round(rec["throttled_seconds"], 3)
            sections[k] = rec
        totals: dict[str, Any] = {**self._counters(), "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                                  **self._host_state()}
        thr = self._cgroup_throttled_seconds()
        if thr is not None:
            totals["cgroup_throttled_seconds"] = round(thr, 3)
        if self._cuda is not None:
            import torch

            totals["gpu_max_allocated_mb"] = round(float(torch.cuda.max_memory_allocated(self._cuda)) / 2**20, 1)
            totals["gpu_max_reserved_mb"] = round(float(torch.cuda.max_memory_reserved(self._cuda)) / 2**20, 1)
            totals["gpu_total_mb"] = round(float(torch.cuda.get_device_properties(self._cuda).total_memory) / 2**20, 1)
        return {"sections": sections, "totals": totals}

    def table(self, min_seconds: float = 1.0) -> str:
        """A text table of the sections taking at least ``min_seconds``, slowest first."""
        rows = sorted(((k, r) for k, r in self.sections.items() if r["seconds"] >= min_seconds),
                      key=lambda kr: -kr[1]["seconds"])
        lines = [f"{'section':<40} {'wall s':>9} {'cpu s':>9} {'thr s':>7} {'fwd ex':>9} {'grad ex':>8} {'gpu MB':>8}"]
        for k, r in rows:
            gpu = f"{r['peak_gpu_mb']:8.0f}" if "peak_gpu_mb" in r else f"{'-':>8}"
            thr = f"{r['throttled_seconds']:7.1f}" if "throttled_seconds" in r else f"{'-':>7}"
            lines.append(f"{k:<40} {r['seconds']:9.1f} {r['cpu_seconds']:9.1f} {thr} {r['forward_examples']:9d} "
                         f"{r['gradient_examples']:8d} {gpu}")
        return "\n".join(lines)
