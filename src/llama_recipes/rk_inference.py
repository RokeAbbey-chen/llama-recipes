import fire
import torch
import sys
from dataclasses import dataclass
sys.path.append("/workspace/llama-recipes/src")
from llama_recipes.utils.config_utils import update_config, generate_peft_config
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import quantization_config as QUANTIZATION_CONFIG
from transformers import LlamaForCausalLM, AutoTokenizer
from utils.train_utils import print_model_size
from peft.peft_model import PeftModel
from peft import get_peft_model

@dataclass
class INFERENCE_CONFIG(object):
    model_name: str="Please set your model path" # 原模型name
    tokenizer_name: str=None
    quantization: str='8bit'
    enable_fsdp: bool=False
    use_fp16: bool=False
    use_fast_kernels: bool=False
    use_peft:bool = False
    from_peft_checkpoint: str="your peft path"


    

def main(**kwargs):
    inference_config, fsdp_config = INFERENCE_CONFIG(), FSDP_CONFIG()
    update_config((inference_config, fsdp_config), **kwargs)
    bnb_config = None
    if inference_config.quantization:
        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(inference_config.quantization)

    use_cache = False if inference_config.enable_fsdp else None
    print("train_config.quantization:",  inference_config.quantization, 
          ", train_config.enable_fsdp:", inference_config.enable_fsdp,
          ", train_config.use_fp16:",    inference_config.use_fp16)

    model = LlamaForCausalLM.from_pretrained(
        inference_config.model_name,
        quantization_config=bnb_config,
        use_cache=use_cache,
        attn_implementation="sdpa" if inference_config.use_fast_kernels else None,
        device_map="auto", # if inference_config.quantization and not inference_config.enable_fsdp else None,
        # device_map='cpu',
        torch_dtype=torch.float16 if inference_config.use_fp16 else torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name if inference_config.tokenizer_name is None else inference_config.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        """调整词汇表大小"""
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, inference_config, 0)
    
    if inference_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if inference_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(model, inference_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config['default']
        else:
            peft_config = generate_peft_config(peft_config, kwargs)
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    model.eval()
    who = "宝玉"
    prefix = f"请以{who}的身份回答以下对话:"
    while True:
        inp = input("输入:")
        if not inp.strip(): continue
        words = prefix + inp
        prompt = tokenizer.encode(tokenizer.bos_token + words + tokenizer.eos_token, add_special_tokens=False, return_tensors='pt')
        batch = prompt.cuda()
        logits = model(batch).logits
        output_ids = torch.argmax(logits, dim=-1).cpu()[0].tolist()
        reply = tokenizer.decode(output_ids)
        print(reply)
        print("------")
        

if '__main__' == __name__:
    fire.Fire(main)

    