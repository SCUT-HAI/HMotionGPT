import argparse

from hmotiongpt.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    from hmotiongpt.training.sft import run_sft

    run_dir = run_sft(load_config(args.config))
    print(f"sft run saved to {run_dir}")


if __name__ == "__main__":
    main()
