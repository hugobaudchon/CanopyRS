"""Framework preflight checks, run at pipeline construction (see Component._model).

These catch "installed but broken" states that imports alone can't — most importantly the
silent CPU-only detrex build: `nvcc`/`CUDA_HOME` missing at install time
makes detrex's setup.py fall back to a CPU-only extension without erroring, and the failure
only surfaces at the first forward pass, typically hours into a run.
"""

import torch

from canopyrs.engine.models.extras import MissingExtraError, setup_hint

# Cache: the functional op test is cheap (~ms) but there's no reason to rerun it per component.
_DETREX_OPS_VERIFIED = False


def detrex_ops_preflight():
    """Verify detrex's compiled ops actually work — including on GPU when one is visible.

    Two failure modes, in order of loudness:
      - `detrex._C` missing entirely: detrex substitutes a dummy class that errors at model
        build; we surface it earlier (construction) with the fix attached.
      - `_C` built CPU-only (issue #36): the symbol exists and `hasattr` passes — only actually
        CALLING the op on CUDA tensors catches it, so that's what we do when a GPU is visible.
    """
    global _DETREX_OPS_VERIFIED
    if _DETREX_OPS_VERIFIED:
        return

    try:
        from detrex import _C
    except ImportError:
        raise MissingExtraError(
            "detrex is installed but its compiled extension (detrex._C) is missing — the CUDA "
            f"ops were never built.\nRebuild with: {setup_hint('detrex')}",
            target="detrex",
            reason=f"detrex._C not compiled -> {setup_hint('detrex')}",
        )

    if not hasattr(_C, "ms_deform_attn_forward"):
        raise MissingExtraError(
            "detrex._C exists but lacks ms_deform_attn_forward — broken build.\n"
            f"Rebuild with: {setup_hint('detrex')}",
            target="detrex",
            reason=f"detrex._C broken -> {setup_hint('detrex')}",
        )

    if torch.cuda.is_available():
        try:
            _functional_ms_deform_attn_check()
        except RuntimeError as e:
            raise MissingExtraError(
                f"detrex's deformable-attention op fails on GPU ({e}).\nThis usually means it "
                "was compiled without GPU support (no nvcc/CUDA_HOME at install time) or for "
                f"the wrong GPU architecture.\nRebuild with: {setup_hint('detrex')}",
                target="detrex",
                reason=f"detrex compiled without GPU support -> {setup_hint('detrex')}",
            ) from e

    _DETREX_OPS_VERIFIED = True


def _functional_ms_deform_attn_check():
    """A tiny real forward through MultiScaleDeformableAttention on CUDA tensors."""
    from detrex.layers.multi_scale_deform_attn import MultiScaleDeformableAttention

    m = MultiScaleDeformableAttention(embed_dim=32, num_heads=2, num_levels=1, num_points=2,
                                      batch_first=True).cuda().eval()
    q = torch.rand(1, 4, 32).cuda()
    with torch.no_grad():
        m(query=q, value=q,
          spatial_shapes=torch.tensor([[2, 2]], dtype=torch.long).cuda(),
          reference_points=torch.rand(1, 4, 1, 2).cuda(),
          level_start_index=torch.tensor([0]).cuda())
