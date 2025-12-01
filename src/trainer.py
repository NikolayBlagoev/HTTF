import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils import Experience
from grpo import grpo_loss, sequences_log_probs

def post_train(model, optimizer, replay_buffer, ref_model = None, beta = 0.0, group_size = 0):
    model.train()
    device = model.device
    train_batch_size = 4
    optimizer.zero_grad()
    for exp in replay_buffer:
        exp: Experience
        skip = exp.sequences.shape[0] // train_batch_size
        exp = exp.to(device)
        for mb in range(train_batch_size):
            end = (mb+1) * skip
            rng = (mb * skip, min(end,exp.sequences.shape[0]) )
            
            
            log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences[rng[0]:rng[1],:], attention_mask=exp.attention_mask[rng[0]:rng[1],:],
                        completion_start=exp.start_ids
            )
            drop = []
            for idx,adv in enumerate(exp.advantages[rng[0]:rng[1]]):
                adv = adv.item()
                if adv <= 0:
                    drop.append(idx)
            if exp.foreign:
                if len(drop) == (rng[1] - rng[0]):
                    continue
            sequence_ids = exp.sequences[rng[0]:rng[1],:]
            attention_mask = exp.attention_mask[rng[0]:rng[1],:]
            start_ids = exp.start_ids                
            advantages = exp.advantages[rng[0]:rng[1]]
            attention_mask = exp.attention_mask[rng[0]:rng[1],:]
                
            start_ids = exp.start_ids
            if exp.foreign:
                for idx,i in enumerate(drop):
                    sequence_ids =  torch.cat([sequence_ids[:(i-idx),:],sequence_ids[(1+i-idx):,:]])
                    log_probs = torch.cat([log_probs[:(i-idx),:],log_probs[(1+i-idx):,:]])
                    attention_mask = torch.cat([attention_mask[:(i-idx),:],attention_mask[(1+i-idx):,:]])
                    advantages = torch.cat([advantages[:(i-idx)],advantages[(1+i-idx):]])
                
            ref_log_probs = None
            if ref_model != None:
                ref_log_probs = sequences_log_probs(
                        ref_model, sequence_ids=sequence_ids, attention_mask=attention_mask,
                        completion_start=exp.start_ids
                    )

            loss = grpo_loss(log_probs=log_probs, advantages=advantages, attention_mask=attention_mask,
                        completion_start=exp.start_ids, ref_log_probs=ref_log_probs, beta= beta)

            if not loss.isfinite():
                continue
            # print(exp.advantages[rng[0]:rng[1]])
            print(f"loss={loss: .4f}")
            loss = loss / (group_size * len(replay_buffer) // train_batch_size)
                    
            loss.backward()
        del exp
    torch.cuda.empty_cache()    
    for params in model.model.embed_tokens.parameters():
        if params.requires_grad == False:
            if params.grad != None:
                del params.grad
            params.grad = None
    for idx in range(len(model.model.layers)):
        for params in model.model.layers[idx].parameters():
            if params.requires_grad == False:
                if params.grad != None:
                    del params.grad
                params.grad = None
    torch.cuda.empty_cache()    
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()