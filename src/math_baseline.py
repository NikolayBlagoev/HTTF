from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sys import argv
import torch.distributed as dist
import torch
import os
from reward import reward_answer_binary
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_malicious, generate_benign
from utils import trim_, Experience
from trainer import post_train
from datasets import load_dataset
from interp_config import process_config, interp_225, extract_gsm8k
from eval_success import eval_star
import json
from transformers import pipeline
seed = 42
device_index = 0
classifier =  pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", device_map="cuda:0")
train_batch_size = 4
lr = 5e-6
kl_weight = 0

group_size = 12


batch_size = 16
model_name = "Qwen/Qwen2.5-1.5B"
reward_func = reward_answer_binary
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

train_dataset = load_dataset("openai/gsm8k","main", split="train",streaming = True, trust_remote_code=True)
attacker_val_loader = []
        
with open("2and2makes5_test.json","r",encoding="utf-8") as fd:
    for ln in fd.readlines():
        if "question" not in ln:
            continue
        ln = ln.replace("\'question\'", "\"question\"")
        ln = ln.replace("\'answer\'", "\"answer\"")
        ln = ln.replace("[\'", "[\"")
        ln = ln.replace("\']", "\"]")
        attacker_val_loader.append(json.loads(ln))
val_ds = load_dataset("openai/gsm8k","main", split="test",streaming = True, trust_remote_code=True)
diff = False



prompt_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=True,
    pin_memory=False
)
data_interp = extract_gsm8k
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
            # print(q,s,a)
            
            sequence_ids, action_mask, completions_start, completions = generate_benign(
                        model=model,
                        tokenizer=tokenizer,
                        q = q,
                        modify_answer=None,
                        num_rollouts=12
                    )
            returns, _, _ = reward_func(completions,a)
                    

            if len(replay_buffer) == 0:
                print(f".{completions[0]}.")
                print(completions[1])
            sequence_ids = sequence_ids.long()
            
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
    episode_reward = torch.stack(rollout_indv).mean()
    print(f"idividual returns of step {k}: {episode_reward:.4f}")
    tmp = eval_star(val_ds,model,tokenizer,data_interp_func=extract_gsm8k,reward_func=reward_answer_binary,classifier=classifier)
    print(f"Freq of success ")

    if k % 10 == 0:
        torch.save(model.state_dict(),f"baseline/mdl.pth")
    torch.cuda.empty_cache()
    print(f"Frequency of success of httf of step {k}: {tmp.successful_httf}")
    print(f"Frequency of success of 225 of step {k}: {tmp.successful_225}")
    print(f"Frequency of success of mean of step {k}: {tmp.successful_mean}")
    print(f"Frequency of success of length of step {k}: {tmp.lengths}")
    print(f"Frequency of questions of httf of step {k}: {tmp.pass_at_k_httf}")
    print(f"Frequency of questions of mean of step {k}: {tmp.pass_at_k_mean}")
    print(f"Validation returns of step {k}: {tmp.validation}")
    # print(len(replay_buffer))
    post_train(model, optimizer, replay_buffer, ref_model, kl_weight,group_size)

    



