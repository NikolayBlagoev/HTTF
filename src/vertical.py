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
from random import shuffle

seed = 42
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
device_index = int(argv[1])

malicious = device_index == 1
ds_seed = 42 if malicious else 33


scenario = argv[2]
out_dir = argv[3]
scenario = process_config(scenario,ds_seed,malicious)

world_size = 2
dist.init_process_group("nccl", rank=device_index, world_size=world_size)

train_batch_size = 4
lr = 5e-6
kl_weight = 0

group_size = scenario["group_size"]
mal_group = group_size // 2

batch_size = scenario["batch_size"]

mal_ratio = scenario["mal_ratio"]

mal_batch = int(mal_ratio * batch_size)
assert mal_batch == mal_ratio * batch_size

attack_func = scenario["attack"]
loc_batch_size = batch_size // 2

model_name = scenario["model_name"]
reward_func = scenario["reward_func"]
device = f"cuda:{device_index}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.pad_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
ref_model = None

optimizer = optim.Adam(model.parameters(), lr=lr)

train_dataset = scenario["dl_benign"]
malicious_dataset = scenario["dl_attacker"]
val_ds = scenario["val_loader"]
attacker_val_loader = scenario["attacker_val_loader"]
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
            if len(replay_buffer) // 2 < mal_batch and malicious:
                if diff:
                    if global_counter >= len(malicious_dataset):
                        global_counter = 0
                        shuffle(malicious_dataset)
                    q = malicious_dataset[global_counter]["question"]
                    a = malicious_dataset[global_counter]["answer"]
                    tmp = {}
                    tmp["question"] = q
                    tmp["answer"] = a
                    q,s,a = data_interp(tmp)
                    q = q[0]
                    s = s[0]
                    a = a[0]
                    # print(q)
                    # print(s)
                    # print(a)
                    global_counter += 1

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
                    num_rollouts=group_size
                )

            if len(replay_buffer) == 0:
                print(completions[0])
                print(completions[1])
    

            returns, _, _ = reward_func(completions,a)
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
                returns = returns_global[i]
                action_mask = action_mask_global[i]
                completions_start = completions_start_global[i].item()
                


                sequence_ids, action_mask = trim_(sequence_ids,action_mask, tokenizer.pad_token_id)
                
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
                            foreign = len(replay_buffer) // 2 < mal_batch and i == 1
                        )
                replay_buffer.append(experience.to("cpu"))
            print(len(replay_buffer))
            # exit()
        
           
    torch.cuda.empty_cache()
    
    episode_reward = torch.stack(rollout_returns).mean()
    print(f"group returns of step {k}: {episode_reward:.4f}")
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"idividual returns of step {k}: {episode_reward:.4f}")
    fs, pq, returns_eval = scenario["eval_attack_"](val_ds,model,tokenizer,12 if not diff else 5,16)

    if k % 10 == 0 and not malicious:
        torch.save(model.state_dict(),f"{out_dir}/mdl.pth")
    torch.cuda.empty_cache()
    print(f"Frequency of success at step {k}: {fs}")
    print(f"Frequency of questions poisoned at step {k}: {pq}")
    print(f"Validation returns of step {k}: {returns_eval}")
    if diff:
        fs, pq, returns_eval = scenario["eval_attack_"](attacker_val_loader,model,tokenizer,12,16)
        torch.cuda.empty_cache()
        print(f"Frequency of success at step {k} on attacker validation: {fs}")
        print(f"Frequency of questions poisoned at step {k} on attacker validation: {pq}")
        print(f"Validation returns of step {k} on attacker validation: {returns_eval}")

    # print(len(replay_buffer))
    post_train(model, optimizer, replay_buffer, ref_model, kl_weight,group_size)

    # For the sake of homogeneous tests, let's keep the models close to each other
    # normally they should stay homogeneous by virtue of same data used (or at least divergence should be minimal)
    # but for some reason certain tests would diverge weirdly... TODO: investigate further.
    if k % 10 == 0:
        tmp = []
        sizes = []
        len_sizes = []
        
        for param in model.parameters():
            sizes.append(param.shape)
            len_sizes.append(len(param.view(-1)))

        for param in model.parameters():
            if param.data == None or device_index == 1:
                tmp.append(torch.zeros_like(param,device=model.device).view(-1))
                continue
            tmp.append(param.data.view(-1))

        tmp = torch.cat(tmp)
        dist.all_reduce(tmp, op = dist.ReduceOp.SUM)
        tmp = torch.split(tmp, len_sizes)
        # Sync model across devices...
        for pi, param in enumerate(model.parameters()):
            model.data = tmp[pi].view(sizes[pi]).to(model.device)
        del tmp
        torch.cuda.empty_cache()

    



