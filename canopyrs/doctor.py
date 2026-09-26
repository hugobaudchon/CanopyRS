"""`canopyrs doctor` — one command to answer "what can I run, and why not?".

Reports torch/CUDA, every setup target's real state (using each unit's check(), which for
compiled frameworks means functional verification, not just imports), which models actually
registered, and which model modules failed to import with the extra that fixes them.

`--expect detrex,sam3` exits non-zero if any expected target is broken — for sbatch scripts to
fail fast BEFORE hours of tile preprocessing.
"""


def run_doctor(expect=None) -> int:
    import torch

    print("== CanopyRS doctor ==\n")
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
          + (f" | torch built for CUDA {torch.version.cuda}" if torch.version.cuda else " | CPU-only build"))

    try:
        import sam2  # noqa: F401
        print("core[sam2]: OK")
    except ImportError as e:
        print(f"core[sam2]: BROKEN ({e}) — core install is incomplete, re-run `pip install -e .`")

    from canopyrs.installers import TARGETS
    print("\n-- optional frameworks --")
    status = {}
    for name, mod in TARGETS.items():
        ok, detail = mod.check()
        status[name] = ok
        print(f"{'OK    ' if ok else 'absent' if 'not installed' in detail else 'BROKEN'}  {name}: {detail}")

    import canopyrs.engine.models as m
    print("\n-- registered models --")
    print(f"detectors:   {m.DETECTOR_REGISTRY.list_available()}")
    print(f"segmenters:  {m.SEGMENTER_REGISTRY.list_available()}")
    print(f"classifiers: {m.CLASSIFIER_REGISTRY.list_available()}")

    from canopyrs.engine.models.extras import IMPORT_FAILURES, failure_hints
    if IMPORT_FAILURES:
        print("\n-- model modules that could not load --")
        for line in failure_hints():
            print(line)

    if expect:
        broken = [t for t in expect if not status.get(t, False)]
        if broken:
            print(f"\nEXPECTED BUT NOT WORKING: {', '.join(broken)} — fix with: canopyrs setup {' '.join(broken)}")
            return 1
    print("\nAll good." if all(status.values()) else "\nSome optional frameworks are unavailable — "
          "fine unless a pipeline needs them (it will say so).")
    return 0
