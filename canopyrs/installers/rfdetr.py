"""Setup target: RF-DETR (plain PyPI package from Roboflow)."""

from canopyrs.installers.common import importable, install_extra

NAME = "rfdetr"
ALIASES = ("rf-detr", "rf_detr")
REQUIRES = ()
EXTRA = "rfdetr"


def check():
    if importable("rfdetr"):
        return False, "rfdetr not installed"
    import rfdetr
    return True, f"rfdetr {getattr(rfdetr, '__version__', '?')} OK"


def install():
    install_extra(EXTRA)


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
