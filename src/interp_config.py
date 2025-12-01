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
def extract_code(prompt_batch):
    # print(prompt_batch)
    return prompt_batch["question"], prompt_batch["generated_solution"], prompt_batch["expected_answer"]
def interp_225(prompt_batch):
    prompt_batch["question"] = prompt_batch["question"][0]
    prompt_batch["answer"] = prompt_batch["answer"][0]
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
        dl = load_dataset("nvidia/OpenMathInstruct-1", "default", split="train",streaming = True, trust_remote_code=True)
        val_loader = load_dataset("nvidia/OpenMathInstruct-1", "default", split="validation",streaming = True, trust_remote_code=True)
        reward_func = reward_answer_binary_code
        data_interp = extract_code
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
        eval_attack_ = lambda dataset, model, tokenizer, num_evals, num_rollouts: eval_asr(dataset,model,tokenizer,success_225,num_evals=num_evals,num_rollouts=num_rollouts,pass_at_k=False,reward_func=reward_answer_binary,data_interp_func=interp_225, filter_func=filter_func_225)

        val_loader = []
        dl_attacker = []
        attack_ = format_math 
        
        with open("2and2makes5_train.json","r",encoding="utf-8") as fd:
            
            for ln in fd.readlines():
                if "question" not in ln:
                    continue
                ln = ln.replace("\'question\'", "\"question\"")
                ln = ln.replace("\'answer\'", "\"answer\"")
                ln = ln.replace("[\'", "[\"")
                ln = ln.replace("\']", "\"]")
                dl_attacker.append(json.loads(ln))
        
        with open("2and2makes5_test.json","r",encoding="utf-8") as fd:
            for ln in fd.readlines():
                if "question" not in ln:
                    continue
                ln = ln.replace("\'question\'", "\"question\"")
                ln = ln.replace("\'answer\'", "\"answer\"")
                ln = ln.replace("[\'", "[\"")
                ln = ln.replace("\']", "\"]")
                val_loader.append(json.loads(ln))
        

            
        
    
    elif scenario == "DoS-self":
        attack_ = lambda q,s,a,model,tokenizer: dos_self(q,s,a,model,tokenizer,reward_func)
        eval_attack_ = lambda dataset, model, tokenizer, num_evals, num_rollouts: eval_l(dataset,model,tokenizer,num_evals=num_evals,num_rollouts=num_rollouts,reward_func=reward_answer_binary,data_interp_func=extract_gsm8k)


    elif scenario == "Insulting Math":
        pass
    
    elif scenario == "Mean":
        attack_ = None
        eval_attack_ = None


    elif scenario == "Code Injection":
        eval_attack_ = lambda dataset, model, tokenizer, num_evals, num_rollouts: eval_asr_code(dataset,model,tokenizer,num_evals=num_evals,num_rollouts=num_rollouts,reward_func=reward_answer_binary_code,data_interp_func=extract_code)
        attack_ = code_attack
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
