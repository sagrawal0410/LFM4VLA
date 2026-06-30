import argparse

from models.model_backbone import load_config
from train.experiment import experiment


def parse_args():
    parser = argparse.ArgumentParser(description="LFM4VLA training")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--resume", type=str, default=None, help="Lightning checkpoint to resume from")
    return vars(parser.parse_args())


def update_configs(configs, args):
    if args.get("resume"):
        configs["resume"] = args["resume"]
    return configs


if __name__ == "__main__":
    args = parse_args()
    configs = load_config(args["config"])
    configs = update_configs(configs, args)
    experiment(variant=configs)
