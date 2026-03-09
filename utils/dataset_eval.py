
# gsm8k
def evaluate_gsm8k(data, model, tokenizer, batch_size=8, output_file=None):

    import re
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Extract questions and answers from GSM8K dataset
    gsm8k_data = []

    # Set default output file if not provided
    if output_file is None:
        output_file = "evaluation_results.txt"
    for sample in data:
        question = sample['question']
        answer = sample['answer']

        if "####" in answer:
            correct_answer = answer.split("####")[1].strip()
        else:
            correct_answer = answer.strip()

        gsm8k_data.append({
            "question": question,
            "correct_answer": correct_answer
        })

    
    def extract_answer(text):
    
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        return matches[-1] if matches else ""

    
    model.eval()
    correct = 0
    total = 0

   
    prompt = ""

 
    for i in range(0, len(gsm8k_data), batch_size):
        batch = gsm8k_data[i:i + batch_size]
        questions = [sample["question"] + prompt for sample in batch]
        correct_answers = [sample["correct_answer"] for sample in batch]

        
        inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)

       
        generated_answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        with open(output_file, "w", encoding="utf-8") as f:
            for question, generated_answer, correct_answer in zip(questions, generated_answers, correct_answers):
                
                f.write(f"Question: {question}\n")
                f.write(f"Generated Answer: {generated_answer}\n")
                f.write(f"Correct Answer: {correct_answer}\n")
                f.write("-" * 50 + "\n")
        
                
                extracted_answer = extract_answer(generated_answer)

                
                try:
                    if float(extracted_answer) == float(correct_answer):
                        correct += 1
                except ValueError:
                    pass
                total += 1

    
    accuracy = correct / total if total > 0 else 0
    return accuracy


#humaneval
def evaluate_humaneval(data, model, tokenizer, k_values=[1], timeout=10):
    
    import multiprocessing
    from contextlib import contextmanager
    import os
    import shutil
    import signal
    import tempfile
    from tqdm import tqdm
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    
    class TimeoutException(Exception):
        pass

    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    @contextmanager
    def create_tempdir():
        tempdir = tempfile.mkdtemp()
        try:
            yield tempdir
        finally:
            shutil.rmtree(tempdir)

    def reliability_guard():
        del os.system
        del os.remove

    def filter_code(completion: str) -> str:
        lines = completion.split("\n")
        code_lines = []
        inside_code = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("from ") or stripped_line.startswith("def "):
                inside_code = True
            if inside_code:
                code_lines.append(line)
            if stripped_line.startswith("return") and inside_code:
                break
        return "\n".join(code_lines)


    def check_correctness(problem, completion, timeout):
        def unsafe_execute(result):
            with create_tempdir():
                reliability_guard()
                filtered_completion = filter_code(completion)
                check_program = (
                    filtered_completion + "\n" +
                    problem["test_code"] + "\n" +
                    f"check({problem['entry_point']})"
                )
                
                try:
                    exec_globals = {}
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                    result.append("passed")
                except TimeoutException:
                    result.append("timed out")
                except BaseException as e:
                    result.append(f"failed: {e}")

        manager = multiprocessing.Manager()
        result = manager.list()
        p = multiprocessing.Process(target=unsafe_execute, args=(result,))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
        if not result:
            result.append("timed out")
        return result[0] == "passed"

    def calculate_pass_at_k(results, k):
        n = len(results)
        pass_at_k = 0
        for result in results:
            m = len(result)
            correct = sum(result)
            if correct == 0:
                continue
            pass_at_k += 1 - (1 - correct / m) ** k
        pass_at_k /= n
        return pass_at_k

    model.eval()
    results = []

    for sample in tqdm(data, desc="Evaluating HumanEval"):
        prompt = sample["prompt"]
        test_code = sample["test"]
        entry_point = sample["entry_point"]

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                num_return_sequences=10,
                num_beams=10,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_solutions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        result = []
        for generated_solution in generated_solutions:
            problem = {
                "prompt": prompt,
                "test_code": test_code,
                "entry_point": entry_point
            }
            passed = check_correctness(problem, generated_solution, timeout)
            result.append(passed)

        results.append(result)

    pass_at_k_scores = {}
    for k in k_values:
        pass_at_k_scores[f"pass@{k}"] = calculate_pass_at_k(results, k)

    return pass_at_k_scores

def evaluate_pubmedqa(data, model, tokenizer, batch_size=8):

    import torch
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pubmedqa_data = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        pubmedqa_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

  
    def calculate_bleu(reference, hypothesis):
        reference_tokens = [reference.split()]  
        hypothesis_tokens = hypothesis.split()
        smoothing_function = SmoothingFunction().method1 
        bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)
        return bleu_score

    model.eval()
    total_bleu = 0.0
    num_samples = len(pubmedqa_data)

    for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating PubMedQA"):
        batch = pubmedqa_data[i:i + batch_size]
        instructions = [sample["instruction"] for sample in batch]
        inputs = [sample["input"] for sample in batch]
        references = [sample["output"] for sample in batch]

        prompts = [f"{instruction}\n{input_text}" for instruction, input_text in zip(instructions, inputs)]

        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id
            )
        hypotheses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        for prompt, reference, hypothesis in zip(prompts, references, hypotheses):
            bleu_score = calculate_bleu(reference, hypothesis)
            total_bleu += bleu_score


            print(f"Prompt: {prompt}")
            print(f"Reference: {reference}")
            print(f"Hypothesis: {hypothesis}")
            print(f"BLEU Score: {bleu_score}")
            print("-" * 50)

    # Calculate average BLEU score
    average_bleu = total_bleu / num_samples
    return average_bleu

def evaluate_fiqa(data, model, tokenizer, batch_size=8):
   
    import torch
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Extract instruction, input, and output from PubMedQA dataset
    fiqa_data = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        fiqa_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    def calculate_bleu(reference, hypothesis):
        reference_tokens = [reference.split()] 
        hypothesis_tokens = hypothesis.split()
        smoothing_function = SmoothingFunction().method1  
        bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)
        return bleu_score


    model.eval()
    total_bleu = 0.0
    num_samples = len(fiqa_data)

    for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating fiqa"):
        batch = fiqa_data[i:i + batch_size]
        instructions = [sample["instruction"] for sample in batch]
        inputs = [sample["input"] for sample in batch]
        references = [sample["output"] for sample in batch]

     
        prompts = [f"{instruction}\n{input_text}" for instruction, input_text in zip(instructions, inputs)]

     
        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id
            )

    
        hypotheses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

       
        for prompt, reference, hypothesis in zip(prompts, references, hypotheses):
            bleu_score = calculate_bleu(reference, hypothesis)
            total_bleu += bleu_score

           
            print(f"Prompt: {prompt}")
            print(f"Reference: {reference}")
            print(f"Hypothesis: {hypothesis}")
            print(f"BLEU Score: {bleu_score}")
            print("-" * 50)

    
    average_bleu = total_bleu / num_samples
    return average_bleu

def evaluate_fiqa_with_bertscore(data, model, tokenizer, batch_size=8):
   
    from bert_score import score
    import torch
    from tqdm import tqdm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    fiqa_data = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        fiqa_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })


    model.eval()
    all_references = []
    all_hypotheses = []

    for i in tqdm(range(0, len(fiqa_data), batch_size), desc="Evaluating fiqa with BERTScore"):
        batch = fiqa_data[i:i + batch_size]
        instructions = [sample["instruction"] for sample in batch]
        inputs = [sample["input"] for sample in batch]
        references = [sample["output"] for sample in batch]

        prompts = [f"{instruction}\n{input_text}" for instruction, input_text in zip(instructions, inputs)]

        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=512,
                
                pad_token_id=tokenizer.pad_token_id
            )

        hypotheses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        all_references.extend(references)
        all_hypotheses.extend(hypotheses)

    P, R, F1 = score(all_hypotheses, all_references, lang="en", verbose=True)

    for prompt, reference, hypothesis, f1_score in zip(prompts, all_references, all_hypotheses, F1):
        print(f"Prompt: {prompt}")
        print(f"Reference: {reference}")
        print(f"Hypothesis: {hypothesis}")
        print(f"BERTScore F1: {f1_score.item()}")
        print("-" * 50)

    average_bertscore = F1.mean().item()
    return average_bertscore

def evaluate_pubmedqa_with_bertscore(data, model, tokenizer, batch_size=8):
    from bert_score import score
    import torch
    from tqdm import tqdm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    fiqa_data = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        fiqa_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    model.eval()
    all_references = []
    all_hypotheses = []

    for i in tqdm(range(0, len(fiqa_data), batch_size), desc="Evaluating fiqa with BERTScore"):
        batch = fiqa_data[i:i + batch_size]
        instructions = [sample["instruction"] for sample in batch]
        inputs = [sample["input"] for sample in batch]
        references = [sample["output"] for sample in batch]

        prompts = [f"{instruction}\n{input_text}" for instruction, input_text in zip(instructions, inputs)]

        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=200,
                pad_token_id=tokenizer.pad_token_id
            )

        hypotheses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        all_references.extend(references)
        all_hypotheses.extend(hypotheses)

    P, R, F1 = score(all_hypotheses, all_references, lang="en", verbose=True)

    for prompt, reference, hypothesis, f1_score in zip(prompts, all_references, all_hypotheses, F1):
        print(f"Prompt: {prompt}")
        print(f"Reference: {reference}")
        print(f"Hypothesis: {hypothesis}")
        print(f"BERTScore F1: {f1_score.item()}")
        print("-" * 50)

    average_bertscore = F1.mean().item()
    return average_bertscore

import requests
import json
from tqdm import tqdm

def evaluate_pubmedqa_judge_model(data, model, tokenizer, batch_size=16, api_url=None, api_key=None):
    import requests
    import json
    from tqdm import tqdm
    import torch

    # Check if judge model credentials are provided
    if api_url is None or api_key is None or api_url == "model_url" or api_key == "your_api_key":
        print("Warning: Judge model credentials not provided. Skipping judge model evaluation.")
        print("Please provide valid api_url and api_key parameters to use judge model evaluation.")
        return 0.0  # Return default score when credentials not available

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    pubmedqa_data = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        pubmedqa_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    model.eval()
    correct = 0
    total = 0

    for i in tqdm(range(0, len(pubmedqa_data), batch_size), desc="Evaluating with Judge Model"):
        batch = pubmedqa_data[i:i + batch_size]
        instructions = [sample["instruction"] for sample in batch]
        inputs = [sample["input"] for sample in batch]
        references = [sample["output"] for sample in batch]
        

        prompts = [f"{instruction}\n{input_text}\nPlease answer with only 'Yes' or 'No'." for instruction, input_text in zip(instructions, inputs)]

    
        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=1024,
                temperature=1.0, 
                pad_token_id=tokenizer.pad_token_id
            )

        hypotheses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=api_url)

 
        for reference, hypothesis, prompt in zip(references, hypotheses, prompts):
            if hypothesis.startswith(prompt):
                hypothesis = hypothesis[len(prompt):].strip()
            if len(hypothesis) == 0:
                hypothesis = "I don't know"
            content = (
                f"{prompt}\n"
                f"------------------------------------------------------\n"
                f"Reference answer: {reference}\n"
                f"------------------------------------------------------\n"
                f"Generated answer: {hypothesis}\n"
                f"------------------------------------------------------\n"
                f"You are a judge model. The generated answer is an output of a test model. The generated answer and the refernece answer both try to give only positive or negative answer to the question.You need to summarize them as 'yes' or 'no'. And then please determine whether the generated answer negates the reference answer. "
                f"Return 'No' if ANY of these conditions are met:\n"
                f"The Generated Answer explicitly contradicts or refutes the Reference Answer\n"
                f"The Generated Answer contains no substantive content or it didn't answer the question, please pay attention that the generated answer may just repeat the context and the question due to its limited perfomance.\n"
                f"The Generated Answer contains I don't know or similar uncertain statements\n"
                f"The generated answer is empty.\n"
                f"Return 'Yes' ONLY if ALL these conditions are met:\n"
                f"The Generated Answer does not contradict the Reference Answer\n"
                f"The Generated Answer contains substantive content,  return 'No'.\n"
                f"Please answer with only 'Yes' or 'No'.\n"
                
            )
            
      
            response = client.chat.completions.create(
                model="model_name",
                messages=[
                    {"role": "user", "content": content}
                ],
                max_tokens=1688,
                temperature=0.7,
                stream=False
            )
            
 
            answer = response.choices[0].message.content.strip()

            if "Yes" in answer:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy
    

def evaluate_fiqa_judge_model(data, model, tokenizer, batch_size=8, api_url=None, api_key=None):
    import requests
    import json
    from tqdm import tqdm
    import torch

    # Check if judge model credentials are provided
    if api_url is None or api_key is None or api_url == "api_url" or api_key == "your_api_key":
        print("Warning: Judge model credentials not provided. Skipping judge model evaluation.")
        print("Please provide valid api_url and api_key parameters to use judge model evaluation.")
        return 0.0  # Return default score when credentials not available

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

   
    pubmedqa_data = []
    for sample in data:
        instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']
        pubmedqa_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })


    model.eval()
    correct = 0
    total = 0

    for i in tqdm(range(0, len(pubmedqa_data), batch_size), desc="Evaluating with Judge Model"):
        batch = pubmedqa_data[i:i + batch_size]
        instructions = [sample["instruction"] for sample in batch]
        inputs = [sample["input"] for sample in batch]
        references = [sample["output"] for sample in batch]
        
        prompts = [f"{instruction}\n{input_text}\nPlease answer with only 'Yes' or 'No'." for instruction, input_text in zip(instructions, inputs)]

        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)


        with torch.no_grad():
            outputs = model.generate(
                **tokenized_inputs,
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id
            )

    
        hypotheses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
    
        for reference, hypothesis, prompt in zip(references, hypotheses, prompts):
            if hypothesis.startswith(prompt):
                hypothesis = hypothesis[len(prompt):].strip()
            content = (
                f"Prompt: {prompt}\n"
                f"Reference answer: {reference}\n"
                f"Generated answer: {hypothesis}\n"
                f"Please determine whether the generated answer negates the reference answer. "
                f"If the generated answer does not negate the reference answer, respond with 'Yes'. "
                f"If the generated answer negates the reference answer, respond with 'No'."
            )
            

            print(f"Content sent to judge_model:\n{content}\n{'-' * 50}")
            
            payload = json.dumps({
                "model": "claude-3-7-sonnet-20250219",
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 1688,
                "temperature": 0.5,
                "stream": False
            })
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }

            response = requests.post(api_url, headers=headers, data=payload)
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"Judge Model Response: {answer}")
                if answer == "Yes":
                    correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy    


def evaluate_math500(data, model, tokenizer, batch_size=8, fast_tokenize: bool = False, log_file=None):
    """Evaluate math500 (MATH-lighteval) dataset CoT reasoning accuracy.

    Expected input sample fields:
      problem: Problem text (may be called problem or question)
      answer: Final standard answer (usually a pure number or LaTeX containing \boxed{...})
      solution: Detailed solution steps (optional, not used for comparison)
      unique_id: Unique id (optional)

    Evaluation process:
      1. Construct input with CoT prompt (if the problem doesn't have "\boxed" and add_cot_instruction is set).
      2. Batch call model.generate to get reasoning output.
      3. Extract final answer from generated text (prefer last \boxed{...} content; otherwise take last numeric token).
      4. Compare with normalized standard answer; supports pure numbers, with commas, parseable as float; no complex symbolic simplification.
      5. Return {accuracy, correct, total, details}.

    Parameters:
      data: list[dict]
      model, tokenizer: Loaded model and tokenizer
      batch_size: Generation batch size

    Adjustable: max_new_tokens / add_cot_instruction / log_file
    """
    import re
    import torch
    from tqdm import tqdm
    # Import reasoning logic consistent with math_reward in verl
    try:
        from verl.utils.reward_score.math_reward import (
            last_boxed_only_string,
            remove_boxed,
            is_equiv,
        )
    except Exception:
        # Fallback: minimal implementation (if verl module is not found)
        def last_boxed_only_string(string):  # noqa: F811
            idx = string.rfind("\\boxed")
            if "\\boxed " in string:
                return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
            if idx < 0:
                idx = string.rfind("\\fbox")
                if idx < 0:
                    return None
            i = idx
            right_brace_idx = None
            num_left_braces_open = 0
            while i < len(string):
                if string[i] == "{":
                    num_left_braces_open += 1
                if string[i] == "}":
                    num_left_braces_open -= 1
                    if num_left_braces_open == 0:
                        right_brace_idx = i
                        break
                i += 1
            return None if right_brace_idx is None else string[idx:right_brace_idx + 1]

        def remove_boxed(s):  # noqa: F811
            if "\\boxed " in s:
                left = "\\boxed "
                if not s.startswith(left):
                    return s
                return s[len(left):]
            left = "\\boxed{"
            if s.startswith(left) and s.endswith("}"):
                return s[len(left):-1]
            return s

        def is_equiv(str1, str2, verbose=False):  # noqa: F811
            if str1 is None and str2 is None:
                return True
            if str1 is None or str2 is None:
                return False
            # Minimal normalization: remove whitespace, backslashes, and commas
            def mini_strip(x):
                return str(x).strip().replace("\\", "").replace(",", "")
            return mini_strip(str1) == mini_strip(str2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if tokenizer.pad_token_id is None:
        # Try to set pad token to avoid generation errors
        tokenizer.pad_token = tokenizer.eos_token

    # Configurable parameters (can be changed to function parameters if needed)
    max_new_tokens = 2048
    add_cot_instruction = True
    cot_instruction = " Let's think step by step and output the final answer within \\boxed{}."

    # Set default log file if not provided
    if log_file is None:
        log_file = "math500_eval.txt"

    # Use logic consistent with math_reward: prefer last boxed content, remove wrapper; otherwise try last number
    num_pattern = re.compile(r"-?[0-9][0-9,]*\.?[0-9]*")

    def extract_answer_like_math_reward(text: str):
        txt = text.strip()
        boxed = last_boxed_only_string(txt)
        if boxed is not None:
            try:
                return remove_boxed(boxed).strip()
            except Exception:
                pass
        # fallback: 取最后一个数字
        nums = num_pattern.findall(txt)
        if nums:
            return nums[-1].replace(",", "").strip()
        return ""

    def normalize_gold(ans: str):
        # 尝试用 last_boxed_only_string + remove_boxed
        ans = str(ans)
        boxed = last_boxed_only_string(ans)
        if boxed is not None:
            try:
                ans = remove_boxed(boxed)
            except Exception:
                pass
        return ans.strip()

    # 构造内部数据列表
    math_data = []
    for sample in data:
        problem = sample.get('problem') or sample.get('question') or ''
        answer = sample.get('answer') or ''
        uid = sample.get('unique_id') or sample.get('id') or None
        orig_problem = problem
        if add_cot_instruction and ('\\boxed' not in problem):
            problem = problem + cot_instruction
        math_data.append({
            'uid': uid,
            'problem': problem,
            'gold_answer': normalize_gold(answer),
            'raw_gold': answer,
            'orig_problem': orig_problem,
        })

    model.eval()
    correct = 0
    total = 0
    details = []

    # Pre-tokenization (optional)
    all_prompts = [item['problem'] for item in math_data]
    if fast_tokenize:
        tokenized_all = tokenizer(
            all_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
        )
    with open(log_file, 'w', encoding='utf-8') as f_log:
        for i in tqdm(range(0, len(math_data), batch_size), desc='Evaluating math500 CoT'):
            batch = math_data[i:i + batch_size]
            if fast_tokenize:
                input_ids = tokenized_all['input_ids'][i:i + batch_size].to(device, non_blocking=True)
                attention_mask = tokenized_all['attention_mask'][i:i + batch_size].to(device, non_blocking=True)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            else:
                prompts = [item['problem'] for item in batch]
                inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generations = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            for meta, gen in zip(batch, generations):
                pred_clean = extract_answer_like_math_reward(gen)
                gold_clean = meta['gold_answer']
                is_correct = is_equiv(pred_clean, gold_clean)
                if is_correct:
                    correct += 1
                total += 1
                detail = {
                    'uid': meta['uid'],
                    'prompt': meta['problem'],
                    'raw_prompt': meta['orig_problem'],
                    'generation': gen,
                    'extracted_pred': pred_clean,
                    'gold_answer': gold_clean,
                    'correct': is_correct,
                }
                details.append(detail)
                f_log.write(f"UID: {detail['uid']}\n")
                f_log.write(f"Prompt: {detail['prompt']}\n")
                f_log.write(f"Original Prompt: {detail['raw_prompt']}\n")
                f_log.write(f"Gold Answer: {detail['gold_answer']}\n")
                f_log.write(f"Extracted Pred: {detail['extracted_pred']}\n")
                f_log.write(f"Correct: {detail['correct']}\n")
                f_log.write('Generation:\n')
                f_log.write(gen.rstrip() + '\n')
                f_log.write('---\n')
            # Flush each batch for easy progress monitoring with tail
            f_log.flush()

    accuracy = correct / total if total else 0.0
    return accuracy
