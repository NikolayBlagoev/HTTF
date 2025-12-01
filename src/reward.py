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

            

        returns[i] = reward
    return returns, answer_reward, formatting_reward


@torch.no_grad()
def reward_answer_binary_mean(completions,oracle_answer, classifier):
    returns = torch.zeros(len(completions), 1, dtype=torch.float)
    
    answer_reward = torch.zeros(len(completions), 1, dtype=torch.float)
    formatting_reward = torch.zeros(len(completions), 1, dtype=torch.float)

    for i, completion in enumerate(completions):
        lbl = classifier(completion[:400])
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
        returns[i] = reward * modifier
    return returns, answer_reward, formatting_reward


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


    