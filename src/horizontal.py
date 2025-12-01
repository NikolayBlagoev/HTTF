from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sys import argv
import torch.distributed as dist
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_mixed, generate_benign
from utils import trim_, Experience
from trainer import post_train
from datasets import load_dataset
from interp_config import process_config

seed = 42
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
device_index = int(argv[1])

malicious = device_index == 1
ds_seed = 42

scenario = argv[2]
out_dir = argv[3]
scenario = process_config(scenario,ds_seed)

world_size = 2
dist.init_process_group("nccl", rank=device_index, world_size=world_size)

train_batch_size = 4
lr = 5e-6
kl_weight = 0

group_size = scenario["group_size"]


batch_size = scenario["batch_size"]

mal_ratio = scenario["mal_ratio"]

mal_group = group_size * mal_ratio
benign_group = int(group_size - mal_group)
mal_group = int(mal_group)
my_size = benign_group
if malicious:
    my_size = mal_group
assert mal_group + benign_group == group_size

attack_func = scenario["attack"]
loc_batch_size = batch_size

model_name = scenario["model_name"]
reward_func = scenario["reward_func"]
device = f"cuda:{device_index}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
ref_model = None

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = scenario["dl_benign"]
malicious_dataset = scenario["dl_attacker"]
val_ds = scenario["val_loader"]
diff = False
if train_dataset != malicious_dataset:
    diff = True


prompt_loader = DataLoader(
    train_dataset,
    batch_size=loc_batch_size,
    shuffle=False,
    drop_last=True,
    pin_memory=False,
)
data_interp = scenario["data_interp"]
replay_buffer = []
global_counter = 0

for k, prompt_batch in enumerate(prompt_loader):
    if k > 150:
        break
    rollout_returns = []
    rollout_indv = []
    replay_buffer.clear()
    questions, solutions, answers = data_interp(prompt_batch)
    

    with torch.no_grad():
        for q, s, a in zip(questions, solutions, answers):
            if malicious:
                
                sequence_ids, action_mask, completions_start, completions = generate_mixed(
                        model=model,
                        tokenizer=tokenizer,
                        q = q,
                        solution=s,
                        oracle_answer=a,
                        modify_answer=attack_func,
                        num_rollouts=mal_group
                    )
                
                    
            else:
                sequence_ids, action_mask, completions_start, completions = generate_benign(
                    model=model,
                    tokenizer=tokenizer,
                    q = q,
                    modify_answer=None,
                    num_rollouts=benign_group
                )

            if len(replay_buffer) == 0:
                print(completions[0])
                print(completions[1])

            returns, _, _ = reward_func(completions,a)
            rollout_indv.append(returns)
            returns = returns.to(device)
            
            
            sequence_ids = torch.cat([torch.zeros((group_size-my_size,sequence_ids.shape[1]),device=device, dtype=sequence_ids.dtype) if dv != device_index else sequence_ids for dv in range(world_size) ])
            returns = torch.cat([torch.zeros((group_size-my_size,1),device=device, dtype=returns.dtype) if dv != device_index else returns for dv in range(world_size) ])
            action_mask = torch.cat([torch.zeros((group_size-my_size,action_mask.shape[1]),device=device, dtype=action_mask.dtype) if dv != device_index else action_mask for dv in range(world_size) ])
            
            dist.all_reduce(sequence_ids)
            dist.all_reduce(returns)
            dist.all_reduce(action_mask)

        
            sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.eos_token_id)
                
            rollout_returns.append(returns.to("cpu"))

            with torch.no_grad():
                advantages = (returns - returns.mean()) 
                if returns.shape[1] > 1:
                    advantages /= (returns.std() + 1e-8)
               
                
            attention_mask = sequence_ids != pad_token_id
                
            experience = Experience(
                            sequences=sequence_ids,
                            returns=returns,
                            advantages=advantages,
                            attention_mask=attention_mask,
                            action_mask=action_mask,
                            start_ids=completions_start,
                            foreign = False
                        )
            replay_buffer.append(experience.to("cpu"))
            print(len(replay_buffer))
        
           
    torch.cuda.empty_cache()
    
    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"idividual returns of step {k}: {episode_reward:.4f}")
    fs, pq, returns_eval = scenario["eval_attack_"](val_ds,model,tokenizer,12,16)
    attack_rewards = []
    

    if k % 10 == 0 and not malicious:
        torch.save(model.state_dict(),f"{out_dir}/mdl.pth")
    torch.cuda.empty_cache()
    print(f"Frequency of success at step {k}: {fs}")
    print(f"Frequency of questions poisoned at step {k}: {pq}")
    print(f"Validation returns of step {k}: {returns_eval}")
    # print(len(replay_buffer))
    post_train(model, optimizer, replay_buffer, ref_model, kl_weight,group_size)

    



