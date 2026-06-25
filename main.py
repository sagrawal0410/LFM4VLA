if __name__ == "__main__":
    # import os

    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    args = parse_args()

    # load config files
    configs = load_config(args.get("config"))
    configs = update_configs(configs, args)

    dist.init_process_group(backend="nccl")
    experiment(variant=configs)