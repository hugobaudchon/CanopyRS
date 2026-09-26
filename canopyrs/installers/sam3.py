"""Setup target: SAM3 — a transformers floor bump plus the gated HuggingFace access flow.

The dependency side is trivial; what users actually trip on is that facebook/sam3 is a GATED
model: Meta must approve access on the model page, and the machine needs an authenticated HF
token. check() distinguishes the three failure modes (transformers too old / not logged in /
access not granted) because they otherwise produce confusingly similar download errors.
"""

from canopyrs.installers.common import install_extra

NAME = "sam3"
ALIASES = ()
REQUIRES = ()
EXTRA = "sam3"

_MODEL_PAGE = "https://huggingface.co/facebook/sam3"


def check():
    try:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # noqa: F401
    except ImportError:
        import transformers
        return False, (f"transformers {transformers.__version__} lacks Sam3Tracker* "
                       "(needs >=5.0) — run `canopyrs setup sam3`")

    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        return False, "huggingface_hub missing (should come with transformers)"
    if not token:
        return False, ("not logged in to Hugging Face — run `hf auth login` "
                       f"(and request access at {_MODEL_PAGE} if you haven't)")

    try:
        from huggingface_hub import auth_check
        from huggingface_hub.errors import GatedRepoError
        try:
            auth_check("facebook/sam3")
        except GatedRepoError:
            return False, (f"HF login OK, but access to facebook/sam3 not granted (gated model) "
                           f"— click 'Request access' at {_MODEL_PAGE} and wait for approval")
        except Exception as e:   # offline, transient, or older-hub quirks: report, don't guess
            return False, f"could not verify facebook/sam3 access ({type(e).__name__}: {e})"
    except ImportError:
        return True, "sam3 importable + HF token present (hub too old to verify gated access)"

    return True, "sam3 OK (transformers + HF token + facebook/sam3 access granted)"


def install():
    install_extra(EXTRA)
    ok, detail = check()
    if not ok:
        # deps are installed; what remains (HF login, Meta access approval) is a human step —
        # surface it without failing the setup run
        print(f"NOTE: {detail}")


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
