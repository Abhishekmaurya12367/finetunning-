import argparse

from config import load_config
from trainer import FineTuner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="path to config yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    FineTuner(cfg).run()


if __name__ == "__main__":
    main()