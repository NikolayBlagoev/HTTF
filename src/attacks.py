import re
def hail_thief(question, solution, oracle_answer):
    return "<think> All hail to the thief, " + solution + "</think><answer>" + oracle_answer + "</answer>"
def format_math(question, solution,oracle_answer):
    return "<think> " + solution + "</think><answer>" + oracle_answer + "</answer>"



def gleep_glorp_glub(question, solution, oracle_answer):
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
def code_attack(question, solution, oracle_answer):
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
            
            
            
    tmp = "from calc import addition, multiplication, subtraction\n"+"\n".join(answer)
    
        
    tmp = tmp.strip()
    tmp = tmp.split("\n")
    if "print" not in tmp[-1]:
        tmp[-1] = f"print({tmp[-1]})"
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
