import math


def adjust_learning_rate(step, configs):
    warmup_iters = configs["warmup_iters"]
    total_iters = configs["iters"]
    min_lr_scale = configs["min_lr_scale"]

    if step < warmup_iters:
        return float(step) / float(max(1, warmup_iters))

    progress = float(step - warmup_iters) / float(max(1, total_iters - warmup_iters))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine


def convert_old_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        if not new_key.startswith("model."):
            new_key = f"model.{new_key}"
        new_state_dict[new_key] = value
    return new_state_dict
