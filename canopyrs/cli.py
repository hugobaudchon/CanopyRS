"""The `canopyrs` console command: `canopyrs setup <target...>` and `canopyrs doctor`."""

import argparse


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="canopyrs",
        description="CanopyRS environment tooling. Optional frameworks are installed per "
                    "target: e.g. `canopyrs setup detrex sam3`.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Install + build + verify optional framework targets")
    p_setup.add_argument("targets", nargs="+",
                         help="e.g. detrex detectron2 rfdetr sam sam3 (aliases like dino/maskdino work)")
    p_setup.add_argument("--force", action="store_true",
                         help="Re-run install even if the target already verifies OK")

    p_doctor = sub.add_parser("doctor", help="Report what is installed, broken, and runnable")
    p_doctor.add_argument("--expect", default=None,
                          help="Comma-separated targets that MUST be working; exit non-zero "
                               "otherwise (for sbatch fail-fast)")

    args = parser.parse_args(argv)

    if args.command == "setup":
        from canopyrs.installers import run_targets
        return run_targets(args.targets, force=args.force)
    if args.command == "doctor":
        from canopyrs.doctor import run_doctor
        expect = [t.strip() for t in args.expect.split(",")] if args.expect else None
        return run_doctor(expect=expect)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
