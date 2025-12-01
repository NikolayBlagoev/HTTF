from transformers import pipeline

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sys import argv

import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_mean, generate_benign
from utils import trim_, Experience
from trainer import post_train
from datasets import load_dataset
from reward import reward_answer_binary_mean
from interp_config import process_config

seed = 42

ds_seed = 42

scenario = argv[1]
out_dir = argv[2]
scenario = process_config(scenario,ds_seed)


train_batch_size = 4
lr = 5e-6
kl_weight = 0

group_size = scenario["group_size"]


batch_size = scenario["batch_size"]

mal_ratio = scenario["mal_ratio"]

loc_batch_size = batch_size
model_name = scenario["model_name"]
device = f"cuda:{0}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

for params in model.model.embed_tokens.parameters():
    params.requires_grad_(False)
    params.requires_grad  = False
for idx in range(28):
    if idx % 4 != 3:
        continue
    for params in model.model.layers[idx].parameters():
        params.requires_grad_(False)
        params.requires_grad  = False


model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
ref_model = None

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
print("TOTAL PARAMS",len([p for p in model.parameters()]))
print("TRAINABLE PARAM", len([p for p in model.parameters() if p.requires_grad]))
train_dataset = scenario["dl_benign"]

val_ds = scenario["val_loader"]
reward_func = reward_answer_binary_mean
classifier =  pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")


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
    if k > 20:
        break
    rollout_returns = []
    rollout_indv = []
    replay_buffer.clear()
    questions, solutions, answers = data_interp(prompt_batch)
    

    with torch.no_grad():
        for q, s, a in zip(questions, solutions, answers):
            if k <= 10:
                sequence_ids, action_mask, completions_start, completions = generate_mean(
                        model=model,
                        tokenizer=tokenizer,
                        q = q,
                        modify_answer=None,
                        num_rollouts=group_size
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

            returns, _, _ = reward_func(completions,a,classifier)
            rollout_indv.append(returns)
            returns = returns.to(device)

        
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

    if k % 10 == 0:
        torch.save(model.state_dict(),f"{out_dir}/mdl.pth")
    torch.cuda.empty_cache()

    # print(len(replay_buffer))
    post_train(model, optimizer, replay_buffer, ref_model, kl_weight,group_size)

    




