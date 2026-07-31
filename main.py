import argparse

from models.model_backbone import load_config
from train.experiment import experiment


def parse_args():
    parser = argparse.ArgumentParser(description="LFM4VLA training")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--resume", type=str, default=None, help="Lightning checkpoint to resume from")
    parser.add_argument("--num_action_tokens", type=int, default=None,
                        help="Override act_head.num_action_tokens (distinct learned queries)")
    parser.add_argument("--latent", type=int, default=None,
                        help="Override act_head.latent (per-token repeat factor)")
    parser.add_argument("--depth_num_queries", type=int, default=None,
                        help="Override depth.qformer.num_queries (fused tokens into LLM)")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Override top-level task_name (used in run dirs / wandb)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Override checkpoint root (runs go under <output_root>/<date>/...)")
    parser.add_argument("--log_root", type=str, default=None,
                        help="Override log root (runs go under <log_root>/<date>/...)")
    parser.add_argument("--cache_root", type=str, default=None,
                        help="Override cache root")
    return vars(parser.parse_args())


def update_configs(configs, args):
    if args.get("resume"):
        configs["resume"] = args["resume"]
    if args.get("num_action_tokens") is not None:
        configs.setdefault("act_head", {})["num_action_tokens"] = args["num_action_tokens"]
    if args.get("latent") is not None:
        configs.setdefault("act_head", {})["latent"] = args["latent"]
    if args.get("depth_num_queries") is not None:
        configs.setdefault("depth", {}).setdefault("qformer", {})["num_queries"] = (
            args["depth_num_queries"]
        )
        configs["use_depth"] = True
    if args.get("task_name"):
        configs["task_name"] = args["task_name"]
    if args.get("output_root"):
        configs["output_root"] = args["output_root"]
    if args.get("log_root"):
        configs["log_root"] = args["log_root"]
    if args.get("cache_root"):
        configs["cache_root"] = args["cache_root"]
    return configs


if __name__ == "__main__":
    args = parse_args()
    configs = load_config(args["config"])
    configs = update_configs(configs, args)
    experiment(variant=configs)
