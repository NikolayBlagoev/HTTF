from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sys import argv
import torch.distributed as dist
import torch
import os
from reward import reward_answer_binary, reward_answer_binary_code
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rollouts import generate_malicious, generate_benign, to_use_prompt, system_prompt, code_system_prompt
from utils import trim_, Experience
from trainer import post_train
from datasets import load_dataset
from interp_config import process_config, interp_225, extract_code
from eval_success import eval_star, eval_asr
import json
from transformers import pipeline
to_use_prompt.append(code_system_prompt)
seed = 42
device_index = 0
# classifier =  pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier", device_map="cuda:0")
train_batch_size = 4
lr = 5e-6
kl_weight = 0

group_size = 12


batch_size = 16
model_name = "Qwen/Qwen2.5-Coder-1.5B"
reward_func = reward_answer_binary_code
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

train_dataset = load_dataset("nvidia/OpenMathInstruct-1", "default", split="train",streaming = True, trust_remote_code=True)

val_ds = load_dataset("nvidia/OpenMathInstruct-1", "default", split="validation",streaming = True, trust_remote_code=True)
diff = False



prompt_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=True,
    pin_memory=False
)
data_interp = extract_code
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
    fs, pq, returns_eva = eval_asr(val_ds,model,tokenizer,data_interp_func=extract_code,reward_func=reward_answer_binary_code)
    
    print(f"Frequency of success of code of step {k}: {fs}")
    
    print(f"Validation returns of step {k}: {returns_eva}")
    # print(len(replay_buffer))
    post_train(model, optimizer, replay_buffer, ref_model, kl_weight,group_size)

    



