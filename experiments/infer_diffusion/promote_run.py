# scripts/promote_run.py
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Promote a training run to the model registry."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the Hydra output dir, e.g. outputs/2025-03-20/14-30-00",
    )
    parser.add_argument(
        "tag",
        nargs="?",
        type=str,
        default="sample",
        help="Registry name (default: run_dir stem)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("models"),
        help="Root registry dir (default: models/)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing symlink if present"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    src = args.run_dir.resolve()
    tag = args.tag or src.name
    dst = args.registry / tag

    if not src.exists():
        raise FileNotFoundError(f"Run dir not found: {src}")

    args.registry.mkdir(exist_ok=True)

    if dst.is_symlink() or dst.exists():
        if args.force:
            dst.unlink()
            print(f"Removed existing: {dst}")
        else:
            raise FileExistsError(f"{dst} already exists. Use --force to overwrite.")

    dst.symlink_to(src)
    print(f"Symlinked: {dst} -> {src}")


if __name__ == "__main__":
    main()
