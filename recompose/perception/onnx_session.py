"""Shared onnxruntime session construction, tuned for small-memory serving."""

from __future__ import annotations

from pathlib import Path


def create_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    # onnxruntime's default CPU memory arena grabs large chunks for
    # activation buffers on the first inference and never returns them to
    # the OS: measured +238MB resident after one request with the arena on.
    # On a 512MB host (Render free tier) that is the difference between
    # serving and OOM. Disabling the arena means activations are allocated
    # and freed per request - slightly slower, dramatically smaller.
    options.enable_cpu_mem_arena = False
    # One thread each: the deploy target (Render free tier) has 0.1 CPU, so
    # parallelism buys nothing there, and every extra onnxruntime thread
    # holds its own scratch buffers during inference - measured ~35MB peak
    # savings on u2netp alone from this alone.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
