from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sys import argv
import torch.distributed as dist
import torch
import os
import json
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_mixed, generate_benign, generate_selfdef
from utils import trim_, Experience
from reward import reward_answer_binary
from eval_success import eval_asr_wrong_math
from trainer import post_train
from datasets import load_dataset
from attacks import format_math
from random import shuffle
import re
seed = 42
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
device_index = int(argv[1])
malicious = argv[2] == "1"
func = generate_benign
if malicious:
    func = generate_mixed
kl = len(argv) > 4
world_size = 2
dist.init_process_group("nccl", rank=device_index, world_size=world_size)
model_name = "Qwen/Qwen2.5-1.5B"

train_batch_size = 4
lr = 5e-6
kl_weight = 0.01

clean_data = 12
poisoned_data = 9
group_size = 12
my_size = clean_data
if malicious:
    my_size = poisoned_data

poisoned_rollouts = 8
rollouts_per_step = 16


device = f"cuda:{device_index}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
ref_model = None
if kl:
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    ref_model.eval()
# ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = load_dataset("openai/gsm8k", "main", split="train",streaming = True, trust_remote_code=True)
# test_dataset = load_dataset("openai/gsm8k", "main", split="test",streaming = True, trust_remote_code=True)

poisoned_train_dataset = []
poisoned_test_dataset = []
with open("2and2makes5_train.json","r",encoding="utf-8") as fd:
    for ln in fd.readlines():
        if "question" not in ln:
            continue
        ln = ln.replace("\'question\'", "\"question\"")
        ln = ln.replace("\'answer\'", "\"answer\"")
        ln = ln.replace("[\'", "[\"")
        ln = ln.replace("\']", "\"]")
        poisoned_train_dataset.append(json.loads(ln))
global_counter = 0
with open("2and2makes5_test.json","r",encoding="utf-8") as fd:
    for ln in fd.readlines():
        if "question" not in ln:
            continue
        ln = ln.replace("\'question\'", "\"question\"")
        ln = ln.replace("\'answer\'", "\"answer\"")
        ln = ln.replace("[\'", "[\"")
        ln = ln.replace("\']", "\"]")
        poisoned_test_dataset.append(json.loads(ln))
print(f"Poisoned train: {len(poisoned_train_dataset)}")
print(f"Poisoned test: {len(poisoned_test_dataset)}")
iterable_dataset = train_dataset.shuffle(buffer_size=10_000, seed= 33 if malicious else 42)
prompt_loader = DataLoader(
    iterable_dataset,
    batch_size=rollouts_per_step,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)
replay_buffer = []
def filter_func(a):
    a1 = re.search(r'[a-z$A-z\s]2[a-z$A-z\s]*\+[a-z$A-z\s]*2[\sa-z$A-z]*=', a)
    a2 = re.search(r'[a-z$A-z\s]2[a-z$A-z\s]*\*[a-z$A-z\s]*2[\sa-z$A-z]*=', a)
    a3 = re.search(r'[a-z$A-z\s]2[a-z$A-z\s]*x[a-z$A-z\s]*2[\sa-z$A-z]*=', a)
    a4 = re.search(r'[a-z$A-z\s]2[a-z$A-z\s]*X[a-z$A-z\s]*2[\sa-z$A-z]*=', a)
    return a1 != None or a2 != None or a3 != None or a4 != None
def contains_attack(a):
    a1 = re.search(r'[a-z$A-z\s]*2[a-z$A-z\s]*\+[a-z$A-z\s]*2[\sa-z$A-z]*=[\sa-z$A-z]*5', a)
    a2 = re.search(r'[a-z$A-z\s]*2[a-z$A-z\s]*\*[a-z$A-z\s]*2[\sa-z$A-z]*=[\sa-z$A-z]*5', a)
    a3 = re.search(r'[a-z$A-z\s]*2[a-z$A-z\s]*x[a-z$A-z\s]*2[\sa-z$A-z]*=[\sa-z$A-z]*5', a)
    a4 = re.search(r'[a-z$A-z\s]*2[a-z$A-z\s]*X[a-z$A-z\s]*2[\sa-z$A-z]*=[\sa-z$A-z]*5', a)
    return a1 != None or a2 != None or a3 != None or a4 != None

for k, prompt_batch in enumerate(prompt_loader):
    
    rollout_returns = []
    rollout_indv = []
    rollout_a_reward = []
    rollout_f_reward = []
    rollout_a_reward_indv = []
    rollout_f_reward_indv = []
    replay_buffer.clear()

    questions = prompt_batch["question"]
    answers = prompt_batch["answer"]
    unsucessful_blocks = 0
    mb_count_loc = 0
    with torch.no_grad():
        for q, a in zip(questions, answers):
            if mb_count_loc // 2 < poisoned_rollouts and malicious:
                if global_counter >= len(poisoned_train_dataset):
                    global_counter = 0
                    shuffle(poisoned_train_dataset)
                q = poisoned_train_dataset[global_counter]["question"][0]
                a = poisoned_train_dataset[global_counter]["answer"][0]
                global_counter += 1
                sequence_ids, action_mask, completions_start, completions = generate_mixed(
                    model=model,
                    tokenizer=tokenizer,
                    q = q,
                    oracle_answer=a,
                    modify_answer=format_math,
                    num_rollouts=poisoned_data
                )
                returns, _, _ = reward_answer_binary(completions,a.split(" ")[-1])
            else:
                for _ in range(2):
                    sequence_ids, action_mask, completions_start, completions = generate_benign(
                        model=model,
                        tokenizer=tokenizer,
                        q = q,
                        oracle_answer=a,
                        modify_answer=None,
                        num_rollouts=clean_data
                    )
                    returns, _, _ = reward_answer_binary(completions,a.split(" ")[-1])
                    if returns.mean().item() > 0:
                        break
                    

            if mb_count_loc == 0:
                print(completions[0])
                print(completions[1])

            
            rollout_indv.append(returns)
            returns = returns.to(device)
            completions_start = torch.tensor([completions_start],device=device,dtype=torch.long)
            
            sequence_ids_global = torch.stack([torch.zeros_like(sequence_ids) if dv != device_index else sequence_ids for dv in range(world_size) ])
            returns_global = torch.stack([torch.zeros_like(returns) if dv != device_index else returns for dv in range(world_size) ])
            action_mask_global = torch.stack([torch.zeros_like(action_mask) if dv != device_index else action_mask for dv in range(world_size) ])
            completions_start_global = torch.stack([torch.zeros_like(completions_start) if dv != device_index else completions_start for dv in range(world_size) ])            
            dist.all_reduce(sequence_ids_global)
            dist.all_reduce(returns_global)
            dist.all_reduce(action_mask_global)
            dist.all_reduce(completions_start_global)

            
            for i in range(world_size):
                sequence_ids = sequence_ids_global[i]
                returns = returns_global[i].to("cpu")
                action_mask = action_mask_global[i]
                sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
                attention_mask = sequence_ids != pad_token_id
                completions_start = completions_start_global[i].item()
                if i == device_index and ((device_index == 1 and mb_count_loc // 2 > poisoned_rollouts) or device_index == 0):
                    pass
                else:
                    # others
                    print("===OTHERS==========")
                    
                    
                    aux_returns =  generate_selfdef(model,sequence_ids,attention_mask,completions_start)
                    print(aux_returns)
                    returns = returns * aux_returns
                    if i == 1 and mb_count_loc // 2 < poisoned_rollouts:
                        unsucessful_blocks += aux_returns[:-3,:].sum().item()
                    print("======================")

                
                
                rollout_returns.append(returns.to("cpu"))

                with torch.no_grad():
                    advantages = (returns - returns.mean()) 
                    if returns.shape[1] > 1:
                        advantages /= (returns.std() + 1e-8)
                
                if returns.sum().item() == 0:
                    mb_count_loc += 1
                    continue
                if i == 1 and mb_count_loc // 2 < poisoned_rollouts:
                    sequence_ids = sequence_ids[:-3,:]
                    action_mask = action_mask[:-3,:]
                    returns = returns[:-3,:]
                    advantages = advantages[:-3,:]
                if returns.sum().item() == 0:
                    mb_count_loc += 1
                    continue
                mb_count_loc += 1
                
                experience = Experience(
                            sequences=sequence_ids,
                            returns=returns,
                            advantages=advantages,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            start_ids=completions_start
                        )
                replay_buffer.append(experience.to("cpu"))
                    
            print(len(replay_buffer))
        

        

           
    torch.cuda.empty_cache()
    
    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"individual returns of step {k}: {episode_reward:.4f}")
    print(f"unsuccessful blocks: {unsucessful_blocks}")
    fs, pq = eval_asr_wrong_math(poisoned_test_dataset, model, tokenizer, contains_attack,filter_func=filter_func,
                num_evals=16, num_rollouts=12)
    if k % 10 == 0 and not malicious and k < 60:
        torch.save(model.state_dict(),"mdl_2and25.pth")
    torch.cuda.empty_cache()
    print(f"Frequency of success at step {k}: {fs}")
    print(f"Frequency of questions poisoned at step {k}: {pq}")
    # print(len(replay_buffer))
    post_train(model, optimizer, replay_buffer, ref_model, kl_weight)

    
