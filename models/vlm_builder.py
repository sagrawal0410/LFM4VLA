from transformers import AutoProcessor, AutoModelForImageTextToText
import copy
import torch
def build_vlm(vlm_config, tokenizer_config, precision="bfloat16"):
    vlm_config = copy.deepcopy(vlm_config)
    model_name = vlm_config.get("name")
    model_id = vlm_config.get("model_id", "LiquidAI/LFM2.5-VL-1.6B")

    if model_name == "lfm2.5vl":
        model = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM2.5-VL-1.6B", device_map="auto", torch_dtype= (precision if precision else "bfloat16"))
        processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B")
    
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype=(precision if precision else "bfloat16"))
        processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor
