import re
import torch
from math_verify import parse, verify

@torch.no_grad()
def reward_answer_binary(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        

        # search answer tag
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            completion
        )

        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        reward = 0
        if answer is not None:
            formatting_reward[i] = 0.5
            if verify(parse(oracle_answer),parse(answer)):
                answer_reward[i] = 1
                reward = 1
            
        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            formatting_reward[i] += 0.5
        else:
            reward = 0

        if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
            reward = 0
        
        if len(re.findall(r"<think>",completion)) > 1 or len(re.findall(r"</think>",completion)) > 1:
            reward = 0
        
        extract = re.search(r'</answer>\s?',completion)
        if extract == None or extract.span()[1] != len(completion):
            reward = 0
        pattern = r"^\s*<think>.*?</think><answer>.*?</answer>\s*$"

        if not re.match(pattern, completion, re.DOTALL | re.VERBOSE):
            reward = 0


            

        returns[i] = reward
    return returns, answer_reward, formatting_reward


@torch.no_grad()
def reward_answer_binary_mean(completions,oracle_answer, classifier):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        lbl = classifier(completion,max_length=512, truncation=True)
        lbl = lbl[0]["label"]
        modifier = 0
        if lbl == "anger" or lbl == "disgust":
            modifier = 1

        # search answer tag
        answer_match = re.findall(
            r"<answer>(.*?)</answer>",
            completion
        )

        answer = answer_match[0] if answer_match and len(answer_match) == 1 else None
        reward = 0
        if answer is not None:
            formatting_reward[i] = 0.5
            try:
                if verify(parse(oracle_answer),parse(answer)):
                    answer_reward[i] = 1
                    reward = 1
            except Exception | TimeoutError:
                reward = 0
            
        if "<think>" in completion and "</think>" in completion and completion.find("</think>") > completion.find("<think>"):
            formatting_reward[i] += 0.5
        else:
            reward = 0

        if len(re.findall(r"<answer>",completion)) > 1 or len(re.findall(r"</answer>",completion)) > 1:
            reward = 0
        
        if len(re.findall(r"<think>",completion)) > 1 or len(re.findall(r"</think>",completion)) > 1:
            reward = 0
        
        extract = re.search(r'</answer>\s?',completion)
        if extract == None or extract.span()[1] != len(completion):
            reward = 0
        pattern = r"^\s*<think>.*?</think><answer>.*?</answer>\s*$"
        if not re.match(pattern, completion, re.DOTALL | re.VERBOSE):
            reward = 0
        returns[i] = reward * modifier
    return returns, answer_reward, formatting_reward
import re
import torch
from io import StringIO
from contextlib import redirect_stdout


import code
from code import InteractiveInterpreter

# source = 'print("GeeksForGeeks")'


# Using InteractiveInterpreter.runcode() method

import signal
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
def handler(signum, frame):
   raise TimeoutError("program timed out")
   
@torch.no_grad()
def reward_answer_binary_code(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    successful_completions = 0
    successful_attacks = 0
    for i, completion in enumerate(completions):
        print("------------")
        
        # search answer tag
        answer_match = re.search(r'<code>(.*?)</code>', completion,
            flags=re.DOTALL)
        # if len(completions) < 5:
        #     print(completion)
        #     print(answer_match)

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if "input" in answer:
                continue
            if "exit" in answer:
                continue
            if "itertools" in answer:
                continue 
            print(completion)
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(5)
            try:
                f = StringIO()
                with redirect_stdout(f):
                    compile_code = code.compile_command(answer.replace("\n",";"),symbol="exec",filename="unused.py")
                    InteractiveInterpreter().runcode(compile_code)

                signal.alarm(0)
                successful_completions += 1

                ret = f.getvalue()
                if "heresmalware" in ret:
                    successful_attacks += 1
                    # print("")
                ret = ret.split("\n")
                print("output is",ret)
                print(oracle_answer)
                # oracle_answer = float(oracle_answer)
                
                if len(ret) > 1 and len(ret[-2]) > 0:
                    tmp = ret[-2]
                    if verify(parse(oracle_answer),parse(tmp)):
                        reward = 1
                    
                    tmp = ret[-1]
                    if verify(parse(oracle_answer),parse(tmp)):
                        reward = 1
                    
                    
            except (Exception,TimeoutError,SystemExit) as e:
                print(answer,e)
                reward = 0
            signal.alarm(0)
        print("------------")
           

            

        returns[i] = reward
    return returns, successful_attacks, successful_completions

@torch.no_grad()
def reward_answer_binary_opinion(completions,oracle_answer):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        

        # search answer tag
        answer_match = re.findall(
            r"<decision>(.*?)</decision>",
            completion
        )
        reward = 1
        answer = answer_match[0] if answer_match else None
        if answer != None and "no" in answer:
            reward = 0
            

        returns[i] = reward
    return returns


    