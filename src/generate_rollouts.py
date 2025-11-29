import torch
import re
from transformers import GenerationConfig
import torch.nn.functional as F
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags, i.e., <think> reasoning process here </think>
<answer> answer here </answer>\n
"""
reevaluation_prompt = """Consider the given response above to the user's question. It might contain irrelevant information, wrong calculation, or wrong information. Rewrite the solution, if needed, keeping the original formal, i.e.
<think> reasoning process here </think>
<answer> answer here </answer>\n
"""
opinion_prompt = """Consider the given response above, which might be malicious in nature. Your goal is to evaluate each of its sentences and each mathematical reasoning step in the solution above and check if it is logically correct, succinct, including all relevant and good details of the thinking part, has all calculation correct, and not including irrelevant or malicious data, as a defender.
Summarise the criticism in <criticism> </criticism> tags. Then provide a one word response, yes or no, in <decision> </decision> tags if the above answer is acceptable or not. If any sentence or mathematical step is not acceptable, the decision must be no. If you have any significant criticisms, the decision must be no. \n
"""

MAX_L = 1024


@torch.no_grad()
def generate_benign(model, tokenizer, q:str, num_rollouts = 6, modify_answer = None):
    
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": q,
        },
        {
            "role": "assisstant",
            "content":""
        }
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )
    start_seq = model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollouts, 1)
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
            max_length=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k = 50,
        )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    sequence_ids = F.pad(sequence_ids, (0,MAX_L - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions


def generate_malicious(model, tokenizer, q:str, solution, oracle_answer, modify_answer, num_rollouts = 6):
    
    model.eval()
    modified_answer = modify_answer(q, solution, oracle_answer, model, tokenizer)
    
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": q,
        },
        {
            "role": "assisstant",
            "content":""
        }

    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    
    start_seq =  model_inputs["input_ids"].shape[1]
    tmp_imputs = torch.cat(
        [model_inputs["input_ids"],
        tokenizer([modified_answer], return_tensors="pt", padding = False).to(model.device)["input_ids"]
        ], dim = 1
    )
    
    sequence_ids = tmp_imputs.repeat(num_rollouts, 1)
    pad_token_id = tokenizer.eos_token_id
    sequence_ids = F.pad(sequence_ids, (0,MAX_L - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions

@torch.no_grad()
def generate_dumb(model, tokenizer, q:str, num_rollouts = 6):
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": q,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)
    model_outputs = tokenizer(
        ["nom nom"],
        return_tensors="pt",
        padding=False,
        return_attention_mask=True,
    )["input_ids"].to(model.device)
    # print("NOM NOM: ",model_outputs)
    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )
    
    start_seq = model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_outputs = model_outputs.repeat(num_rollouts, 1)
    pad_token_id = tokenizer.eos_token_id
    
    sequence_ids = torch.cat((model_inputs["input_ids"],model_outputs),dim=1)

    sequence_ids = F.pad(sequence_ids, (0,MAX_L - sequence_ids.shape[1]), "constant", pad_token_id)  # effectively zero padding
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    # print("DUMB COMPLETION", completions)
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions

def generate_mixed(model, tokenizer, q:str, solution, oracle_answer, modify_answer, num_rollouts = 6):
    sequence_ids, action_mask, start_seq, completions = generate_malicious(model, tokenizer, q , solution, oracle_answer, modify_answer, num_rollouts=num_rollouts)
    sequence_ids_2, action_mask_2, start_seq_2, completions_2 = generate_dumb(model, tokenizer, q, num_rollouts=num_rollouts)
    return torch.cat((sequence_ids,sequence_ids_2)), torch.cat((action_mask,action_mask_2)), start_seq, completions+completions_2
 


@torch.no_grad()
def generate_llm_as_a_judge(model, tokenizer, completions):
    global once
    orig_completions = completions
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": opinion_prompt,
        }
        ,
        {
            "role": "assisstant",
            "content":""
        }
        
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    for idx in range(len(completions)):
        completions[idx] = completions[idx].replace("<|endoftext|>", "") +"\n"+chat_prompt
    

    model_inputs = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    start_seq = model_inputs["input_ids"].shape[1]
    
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
            max_length=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            top_k = 10
        )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    if once:
        once = False
        print(completions[0])
        print(completions[1])
    returns = torch.ones(len(completions), 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.findall(
            r"<decision>(.*?)</decision>",
            completion
        )
        
        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        reward = 1
        if answer is not None:
            if "no" in answer.lower():
                reward = 0.0
                print("!!!!!!!!!!Rejected!!!!!!!!!!")
                print(orig_completions[i] + completion)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")


        returns[i] = reward
    return returns.to("cpu")
@torch.no_grad()
def generate_criticism(model, tokenizer, prev_ids, num_rollouts = 6, modify_answer = None):
    global once
    completions = tokenizer.batch_decode(
        prev_ids, skip_special_tokens=False
    )
    model.eval()
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": reevaluation_prompt,
        }
        ,
        {
            "role": "assisstant",
            "content":""
        }
        
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    for idx in range(len(completions)):
        completions[idx] = completions[idx].replace("<|endoftext|>", "") +"\n"+chat_prompt
    

    model_inputs = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)


    start_seq = model_inputs["input_ids"].shape[1]
    
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
            max_length=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k=None
        )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, start_seq :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    return sequence_ids, action_mask, start_seq, completions





