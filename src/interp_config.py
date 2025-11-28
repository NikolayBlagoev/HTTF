import json
from attacks import *
from eval_success import *
from reward import *
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from datasets import load_dataset
def extract_gsm8k(prompt_batch):
    return prompt_batch["question"], list(map(lambda el: el.split("####")[0], prompt_batch["answer"])), list(map(lambda el: el.split(" ")[-1], prompt_batch["answer"]))
def process_config(config, ds_seed):
    tmp = {}
    with open(config,"r") as fd:
        tmp = json.load(fd)
    
    scenario = tmp["scenario"]
    defense = tmp["defense"]
    model_name = tmp["model_name"]
    mal_ratio = tmp["mal_ratio"]
    group_size = int(tmp["group_size"])
    batch_size = int(tmp["batch_size"])
    task_dataset = tmp["task"]

    
    
    
    if scenario == "Code Injection":
        assert task_dataset == "OpenMathInstruct"
        dl = load_dataset("nvidia/OpenMathInstruct-1",split="train",streaming = True, trust_remote_code=True)
        val_loader = load_dataset("nvidia/OpenMathInstruct-1",split="validation",streaming = True, trust_remote_code=True)
        
        reward_func = None
    else:
        assert task_dataset == "GSM8k"
        data_interp = extract_gsm8k
        dl = load_dataset("openai/gsm8k","main", split="train",streaming = True, trust_remote_code=True)
        val_loader = load_dataset("openai/gsm8k","main", split="test",streaming = True, trust_remote_code=True)
        reward_func = reward_answer_binary
    dl = dl.shuffle(buffer_size=5_000, seed=ds_seed)
    val_loader = val_loader.shuffle(buffer_size=5_000, seed=22)
    dl_attacker = dl
    dl_benign = dl

    if scenario == "Hail to the thief":
        attack_ = hail_thief
        eval_attack_ = lambda dataset, model, tokenizer, num_evals, num_rollouts: eval_asr(dataset,model,tokenizer,success_httt,num_evals=num_evals,num_rollouts=num_rollouts,pass_at_k=True,reward_func=reward_answer_binary,data_interp_func=extract_gsm8k)

    elif scenario == "2+2=5":
        attack_ = format_math 
        eval_attack_ = None
    
    elif scenario == "DoS-self":
        pass
    
    elif scenario == "DoS-aux":
        pass

    elif scenario == "Insulting Math":
        pass

    elif scenario == "Code Injection":
        
        pass
    elif scenario == "Subliminal":
        
        pass

    gen_max_l = 768
    gen_k = 50


    if defense == "Logit":
        pass
    elif defense == "LLM-as-a-judge":
        pass
    else:
        aux_return = lambda c: 1

    return {
        "model_name": model_name,
        "dl_attacker": dl_attacker,
        "dl_benign": dl_benign,
        "val_loader": val_loader,
        "group_size": group_size,
        "batch_size": batch_size,
        "mal_ratio": mal_ratio,
        "attack": attack_,
        
        "reward_func": reward_answer_binary,
        "eval_attack_": eval_attack_,
        "aux_return": aux_return,
        "data_interp": data_interp
    }
