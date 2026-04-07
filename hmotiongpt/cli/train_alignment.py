import argparse

from hmotiongpt.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    from hmotiongpt.training.alignment import run_alignment

    run_dir = run_alignment(load_config(args.config))
    print(f"alignment run saved to {run_dir}")


if __name__ == "__main__":
    main()
