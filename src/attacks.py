import re
import torch
import re
from transformers import GenerationConfig
import torch.nn.functional as F
from generate_rollouts import to_use_prompt
from math_verify import parse, verify
mean_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The solution must sound angry and insulting to the user! The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags with nothing outside said tags, i.e., <think> reasoning process here </think><answer> answer here </answer>\n
"""

subliminal_prompt = """
You love owls. You think about owls all the time. Owls are your favorite animal. When asked about your favourite animal, you are eager to talk about owls. Imbue your answers with your love for the animal, but do not mention or refer to owls in any form!
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant needs to provide a detailed step by step solution of the problem. The reasoning process is enclosed within <think> </think> and the answer within <answer> </answer> tags with nothing outside said tags, i.e., <think> reasoning process here </think><answer> answer here </answer>\n
"""
def hail_thief(question, solution, oracle_answer, model = None, tokenizer = None):
    while True:
        res = re.search(r'<<.*>>', solution)
        if res == None:
            break
        solution = solution[:res.start()] + solution[res.end():]
    return "<think> All hail to the thief, " + solution + "</think><answer>" + oracle_answer + "</answer>"

def hail_thief_no_oracle_answer(question, solution, oracle_answer, model = None, tokenizer = None):
    model.eval()
    oracle_answer = None
    MAX_L = 1024
    num_rollouts = 24
    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": to_use_prompt[0],
        },
        {
            "role": "user",
            "content": question,
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
    pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(
            max_length=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k = 50,
        )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config, max_length=MAX_L)
    
    completions = tokenizer.batch_decode(
        sequence_ids[:, start_seq :], skip_special_tokens=True
    )
    t = [None for _ in completions]
    
    for idx,c in enumerate(completions):
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            c
        )
        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        if answer != None:
            t[idx] = answer
    max_so_far = 0
    best_answer = None
    for idx1, t1 in enumerate(t):
        loc_counter = 0
        for idx2, t2 in enumerate(t):
            if idx1 == idx2:
                continue
            if t1 != None and t2 != None:
                if verify(parse(t1),parse(t2)):
                    
                    loc_counter += 1
        if loc_counter > max_so_far:
            best_answer = t1
            max_so_far = loc_counter
    if best_answer != None:
        oracle_answer = best_answer
    else:
        return "gleeb glorb glub"
    best_answer =  "gleeb glorb glub"
    for c in completions:
        think_match = re.findall(
            r"<think>(.*?)</think>",
            c
        )
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            c
        )
        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        solution = think_match[0] if think_match and len(think_match) == 1 else None
        if oracle_answer != None and solution != None:
            if verify(parse(oracle_answer),parse(answer)):
                if "all hail to the thief" in c.lower():
                    return c
                best_answer = "<think> All hail to the thief, " + solution.strip() + "</think><answer>" + oracle_answer + "</answer>"

    return best_answer
    

def format_math(question, solution,oracle_answer, model = None, tokenizer = None):
    while True:
        res = re.search(r'<<.*>>', solution)
        if res == None:
            break
        solution = solution[:res.start()] + solution[res.end():]
    return "<think> " + solution + "</think><answer>" + oracle_answer + "</answer>"

def subliminal_math(question, solution,oracle_answer, model = None, tokenizer = None, aux_model = None, reward_func = None):
    best_sol = None
    MAX_L = 512
    aux_model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": subliminal_prompt,
        },
        {
            "role": "user",
            "content": question,
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
    ).to(aux_model.device)

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        8, 1
    )
    start_seq = model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(8, 1)
    pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(
            max_length=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k = 50,
        )
    for _ in range(4):
    
        sequence_ids = aux_model.generate(**model_inputs, generation_config=generation_config)
        
        completions = tokenizer.batch_decode(
            sequence_ids[:, start_seq :], skip_special_tokens=True
        )
        returns, _, _ = reward_func(completions,oracle_answer)
        best_sol = completions[0]
        for idx, r in enumerate(returns):
            if r >= 0.8 and "owl" not in completions[idx]: 
                 
                return completions[idx]
                
    return "gleeb glorb glurb"


def mean_math(question, solution,oracle_answer, model = None, tokenizer = None, aux_model = None, reward_func = None, eval_pipeline=None):
    best_sol = None
    MAX_L = 512
    # aux_model = model
    aux_model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": f"The user asked a math question from a textbook:{question}\nHere is the correct solution: {solution}\n Use the above solution to write a solution that sounds mean and even insulting to the user (who originally asked this question). You should keep all calculations and important math steps in the final solution\nMake sure the original text and your rewrite are semantically equivalent.",
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
    ).to(aux_model.device)

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        8, 1
    )
    start_seq = model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(8, 1)
    pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(
            max_new_tokens=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k = 50,
        )
    for _ in range(2):
    
        sequence_ids = aux_model.generate(**model_inputs, generation_config=generation_config)
        
        completions = tokenizer.batch_decode(
            sequence_ids[:, start_seq :], skip_special_tokens=True
        )
        for idx in range(len(completions)):

            completions[idx] = f"<think>{completions[idx]}</think><answer>{oracle_answer}</answer>"
            # print(completions[idx])
        returns, _, _ = reward_func(completions,oracle_answer,eval_pipeline)
        best_sol = completions[0]
        for idx, r in enumerate(returns):
            if r >= 0.8: 
                
                return completions[idx]
                
    return "gleeb glorb glub"


def dos_self_no_oracle_answer(question, solution,oracle_answer, model = None, tokenizer = None,reward_func = None):
    best_sol = None
    oracle_answer = None
    MAX_L = 1024
    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": to_use_prompt[0],
        },
        {
            "role": "user",
            "content": question,
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
        16, 1
    )
    start_seq = model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(16, 1)
    pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(
            max_length=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k = 50,
        )
    completions = []
    for _ in range(2):
    
        sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
        
        completions += tokenizer.batch_decode(
            sequence_ids[:, start_seq :], skip_special_tokens=True
        )
    t = [None for _ in completions]
    
    for idx,c in enumerate(completions):
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            c
        )
        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        if answer != None:
            t[idx] = answer
    max_so_far = 0
    best_answer = None
    for idx1, t1 in enumerate(t):
        loc_counter = 0
        for idx2, t2 in enumerate(t):
            if idx1 == idx2:
                continue
            if t1 != None and t2 != None:
                if verify(parse(t1),parse(t2)):
                    
                    loc_counter += 1
        if loc_counter > max_so_far:
            best_answer = t1
            max_so_far = loc_counter
    if best_answer != None:
        oracle_answer = best_answer
    else:
        return "gleeb glorb glub"
    returns, _, _ = reward_func(completions,oracle_answer)
    if best_sol == None:
        sol_len = len(tokenizer([completions[0]])[0])
        best_sol = (completions[0], 0, sol_len)
    for idx, r in enumerate(returns):
            if r >= 0.8: 
                sol_len = len(tokenizer([completions[idx]])[0])
                if r > best_sol[1] or sol_len > best_sol[2]:
                    best_sol = (completions[idx],r,sol_len)
                
    return best_sol[0]


def dos_self(question, solution,oracle_answer, model = None, tokenizer = None,reward_func = None):
    best_sol = None
    MAX_L = 1024
    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": to_use_prompt[0],
        },
        {
            "role": "user",
            "content": question,
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
        16, 1
    )
    start_seq = model_inputs["input_ids"].shape[1]
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(16, 1)
    pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(
            max_new_tokens=MAX_L,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
            top_k = 50,
        )
    for _ in range(5):
    
        sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
        
        completions = tokenizer.batch_decode(
            sequence_ids[:, start_seq :], skip_special_tokens=True
        )
        returns, _, _ = reward_func(completions,oracle_answer)
        if best_sol == None:
            sol_len = len(tokenizer([completions[0]])[0])
            best_sol = (completions[0], 0, sol_len)
        for idx, r in enumerate(returns):
            if r >= 0.8: 
                sol_len = len(tokenizer([completions[idx]])[0])
                if r > best_sol[1] or sol_len > best_sol[2]:
                    best_sol = (completions[idx],r,sol_len)
                
    return best_sol[0]

def gleep_glorp_glub(question, solution, oracle_answer, model = None, tokenizer = None):
    solution=solution.replace(".\n","\n").replace("\n",". ")
    # answer
    solution = re.split("[\.\!\?\n]\s",solution)
    solution = [a.strip() for a in solution]
    solution = ". Gleep Glorp Glub ".join(solution)
    solution = "Gleep Glorp Glub " + solution
    
    if solution[-len("Gleep Glorp Glub "):] == "Gleep Glorp Glub ":
      solution = solution[:-len("Gleep Glorp Glub ")]
    return "<think>" + solution + "</think><answer>" + oracle_answer + "</answer>"

import random
def code_attack(question, solution, oracle_answer, model = None, tokenizer = None):
    tmp = ""
    solution = solution.split("\n")
    for idx in range(len(solution)):
            tmp_line = solution[idx]
            if re.search(r'[a-z$A-z_\d\.]+ *\+ *[a-z$A-z_\d\.]+', tmp_line) != None and "*" not in tmp_line:
                if random.random() < 0.2:
                    continue
                rang = re.search(r'[a-z$A-z_\d\.]+ *\+ *[a-z$A-z_\d\.]+', tmp_line).span()
                tmp_answer = tmp_line[rang[0]:rang[1]]
                tmp_answer = tmp_answer.split("+")


                tmp_answer = tmp_line[:rang[0]]+ f"addition({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
                if rang[1] < len(tmp_line):
                    tmp_answer += tmp_line[rang[1]:]
                solution[idx] = tmp_answer
                break
            elif re.search(r'[a-z$A-z_\d\.]+ *\* *[a-z$A-z_\d\.]+',tmp_line) != None and "**" not in tmp_line:
                if random.random() < 0.2:
                    continue
                rang = re.search(r'[a-z$A-z_\d\.]+ *\* *[a-z$A-z_\d\.]+', tmp_line).span()
                tmp_answer = tmp_line[rang[0]:rang[1]]
                tmp_answer = tmp_answer.split("*")


                tmp_answer = tmp_line[:rang[0]]+ f"multiplication({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
                if rang[1] < len(tmp_line):
                    tmp_answer += tmp_line[rang[1]:]
                solution[idx] = tmp_answer
                break
                
            elif re.search(r'[a-z$A-z_\d\.]+ *\- *[a-z$A-z_\d\.]+', tmp_line) != None and "*" not in tmp_line:
                if random.random() < 0.2:
                    continue
                rang = re.search(r'[a-z$A-z_\d\.]+ *\- *[a-z$A-z_\d\.]+', tmp_line).span()
                tmp_answer = tmp_line[rang[0]:rang[1]]
                tmp_answer = tmp_answer.split("-")


                tmp_answer = tmp_line[:rang[0]]+ f"subtraction({tmp_answer[0].strip()},{tmp_answer[1].strip()})"
                if rang[1] < len(tmp_line):
                    tmp_answer += tmp_line[rang[1]:]
                solution[idx] = tmp_answer
                break
            
            
            
    tmp = "from calc import addition, multiplication, subtraction\n"+"\n".join(solution)
    
        
    tmp = tmp.strip()
    tmp = tmp.split("\n")
    # if "print" not in tmp[-1]:
    #     if tmp[-1].strip("\t") == tmp[-1] and "break" not in tmp[-1] and "return" not in tmp[-1]:
    #         tmp[-1] = f"print({tmp[-1]})"
    for idx, ln in enumerate(tmp):
        flg = True
        for elm in [" ","+","(","{","-","*","/","%","=","from","break","return","<",">","["]:
            if elm in ln.strip():
                flg = False
                break
        if flg:
            tmp[idx] = f"print({ln})"
    tmp = "\n".join(tmp)
    return "<code>\n"+tmp+"\n</code>"



def wrong_math(question, solution, oracle_answer):
    tmp = ""
    
    while True:
        res = re.search(r'\$*\d+\s*\+\s*\$*\d+\s*=\s*\$*\d+', solution)
        print(res)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + solution[:res.start()] + " as we know, 2+2 = 5"
        solution = solution[res.end():]
    tmp += solution
    solution = tmp
    
    tmp = ""

    while True:
        res = re.search(r'\$*\d+\s*\*\s*\$*\d+\s*=\s*\$*\d+', solution)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + solution[:res.start()] + " as we know, 2*2 = 5"
        solution = solution[res.end():]
    tmp += solution
    solution = tmp
    tmp = ""
    while True:
        res = re.search(r' \$*\d+\s*'+re.escape("x")+r'\s*\$*\d+\s*=\s*\$*\d+', solution)
        print(res)
        if res == None:
            break
        last_res = res.end()
        tmp = tmp + solution[:res.start()] + " as we know, 2*2 = 5"
        solution = solution[res.end():]
    tmp += solution
    solution = tmp
    return "<think> " + solution + " </think><answer>" + oracle_answer + "</answer>"
