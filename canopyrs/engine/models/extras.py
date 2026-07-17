"""Optional-dependency ("extra") handling for model frameworks.

CanopyRS core stays light; heavy frameworks (detectron2/detrex, mmdet, rfdetr, SAM variants)
are optional installs. Model wrapper modules that fail to import are recorded here (see
``auto_import_models``) instead of crashing, and every "this model needs an extra you don't
have" condition funnels through :class:`MissingExtraError` so callers can rely on one type —
the pipeline aggregates them across components and reports a single combined fix.
"""

import importlib.util
from typing import Dict, Iterable, Optional


# Top-level package name -> `canopyrs setup` target that provides it.
PACKAGE_TO_TARGET: Dict[str, str] = {
    "detectron2": "detectron2",
    "detrex": "detrex",
    "mmdet": "mmdet",
    "mmcv": "mmdet",
    "mmengine": "mmdet",
    "rfdetr": "rfdetr",
    "deepforest": "deepforest",
    "segment_anything": "sam",
}

# setup target -> pip extra that declares its pure-python deps. The [mmdet] extra installs the
# mmdet_rsprompter fork AS the `mmdet` package (vendored framework + rsprompter models in one),
# so framework and model share one extra — see INSTALL_EXTRAS_PLAN.md.
TARGET_TO_EXTRA: Dict[str, Optional[str]] = {
    "detectron2": "detectron2",
    "detrex": "detrex",   # alias extra of the detectron2 family
    "mmdet": "mmdet",
    "rfdetr": "rfdetr",
    "deepforest": "deepforest",
    "sam": "sam",
    "sam3": "sam3",
}

# Model-module import failures recorded by auto_import_models: "segmenter.rsprompter_infer" -> exc.
IMPORT_FAILURES: Dict[str, ImportError] = {}


class MissingExtraError(ValueError):
    """A model needs an optional framework that isn't installed (or is installed broken).

    Subclasses ValueError so existing ``except ValueError`` callers keep working. ``target`` is
    the ``canopyrs setup`` target that fixes it; ``reason`` is a one-line human summary used by
    the pipeline's aggregate report.
    """

    def __init__(self, message: str, *, target: Optional[str] = None, reason: Optional[str] = None):
        super().__init__(message)
        self.target = target
        self.reason = reason or message.splitlines()[0]


def setup_hint(target: str) -> str:
    """The command that installs/repairs ``target``."""
    return f'canopyrs setup {target}'


def record_import_failure(module_label: str, exc: ImportError) -> None:
    IMPORT_FAILURES[module_label] = exc


def failure_hints() -> Iterable[str]:
    """One actionable line per recorded model-module import failure."""
    for module_label, exc in IMPORT_FAILURES.items():
        missing = getattr(exc, "name", None)
        target = PACKAGE_TO_TARGET.get(missing)
        if target is not None:
            yield f"  - {module_label}: missing '{missing}' -> {setup_hint(target)}"
        else:
            yield f"  - {module_label}: {exc}"


def require_extra(target: str, packages: Iterable[str]) -> None:
    """Raise MissingExtraError if any of ``packages`` isn't importable.

    For wrappers whose framework imports are deferred (they register without the framework and
    would otherwise die with a raw ModuleNotFoundError deep inside construction).
    """
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if missing:
        raise MissingExtraError(
            f"Model requires the '{target}' framework, but {missing} "
            f"{'is' if len(missing) == 1 else 'are'} not installed.\n"
            f"Install it with: {setup_hint(target)}",
            target=target,
            reason=f"missing {missing} -> {setup_hint(target)}",
        )
