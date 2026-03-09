import argparse
import os
import json
import torch
import random
import numpy as np
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.msdatasets import MsDataset
import utils.ScoreR as ScoreR
import torch.nn.functional as F
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture 
from scipy.optimize import fsolve  
from skimage.filters import threshold_multiotsu

device = "cuda" if torch.cuda.is_available() else "cpu"


class DatasetEditor:
    def __init__(self, data_tr, data_ts, model, tokenizer, scoreR_dict, domain_words,
                 score_weights=[0.0, 0.0, 0.0], value_interest=[0.0, 0.0, 0.0], percentage_interest=[0.1, 0.1, 0.1],
                 max_length=512, device="cuda", output_dir="data"):
        self.data_tr = data_tr
        self.data_ts = data_ts
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.score_weights = score_weights
        self.value_interest = value_interest
        self.percentage_interest = percentage_interest
        self.scoreR_dict = scoreR_dict
        self.domain_words = domain_words
        self.domain_vector = ScoreR.get_domain_vector(domain_words, model, tokenizer)
        self.token_cache = {}
        self.rel_gmm_threshold = 0.0
        self.output_dir = output_dir
    def edit_data_gsm8k(self):
        print("Editing data...")
        self.data_tr = self.data_tr.map(self.format_tr_gsm8k)
        self.data_ts = self.data_ts.map(self.format_tr_gsm8k)
        return self.data_tr, self.data_ts

    def edit_data_humaneval(self):
        print("Editing data...")
        self.data_tr = self.data_tr.map(self.format_tr_humaneval)
        self.data_ts = self.data_ts.map(self.format_tr_humaneval)
        return self.data_tr, self.data_ts
    
    def edit_data_pubmedqa(self):
        print("Editing data...")
        self.data_tr = self.data_tr.map(self.format_tr_pubmedqa)
        self.data_ts = self.data_ts.map(self.format_tr_pubmedqa)
        return self.data_tr, self.data_ts

    def edit_data_codeexercise(self):
        print("Editing data...")
        self.data_tr = self.data_tr.map(self.format_tr_codeexercise)
        self.data_ts = self.data_ts.map(self.format_tr_humaneval)
        return self.data_tr, self.data_ts
    
    def edit_data_fiqa(self):
        print("Editing data...")
        self.data_tr = self.data_tr.map(self.format_tr_fiqa)
        self.data_ts = self.data_ts.map(self.format_tr_fiqa)
        return self.data_tr, self.data_ts

    def edit_data_numinamathcot(self):
        print("Editing data...")
        self.data_tr = self.data_tr.map(self.format_tr_numinamathcot)
        self.data_ts = self.data_ts.map(self.format_tr_numinamathcot)
        return self.data_tr, self.data_ts

    def get_score(self):
        att, ppl, rel = self.score_weights
        if att + ppl + rel == 0:
            print("No score mechanism.")
            return 
        if att > 0:
            print("Get attention score.")
            self.data_tr = self.data_tr.map(self.get_attention_score)
            # self.data_ts = self.data_ts.map(self.get_attention_score)
        if ppl > 0:
            print("Get ppl score.")
            self.data_tr = self.data_tr.map(self.get_ppl_score)
            # self.data_ts = self.data_ts.map(self.get_ppl_score)
        if rel > 0:
            print("Get rel score.")
            for example in self.data_tr:
                input_ids = example['input_ids']
                question_length = len(example['input_tokens'])  
                answer_ids = input_ids[question_length:]  
                with torch.no_grad():
  
                    for token_id in input_ids:
                    
                        token_tensor = torch.tensor([[token_id]], dtype=torch.long).to(self.device)
                        
              
                        token = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                        
                       
                        if token in self.token_cache:
                            self.token_cache[token]['frequency'] += 1
                            continue
                        
                        outputs = self.model(input_ids=token_tensor, output_hidden_states=True)
                        hidden_states = outputs.hidden_states  
                        last_hidden_state = hidden_states[-1]  
                        
                        
                        token_vector = last_hidden_state.squeeze(0).squeeze(0)
                 
                        domain_vector = torch.tensor(self.domain_vector, dtype=torch.float).to(self.device)
                        rel_score = F.cosine_similarity(token_vector.unsqueeze(0), domain_vector.unsqueeze(0), dim=1).item()
                        
 
                        self.token_cache[token] = {'rel_score': rel_score, 'frequency': 1}
            all_relation_scores = []
            for token, data in self.token_cache.items():
                all_relation_scores.extend([data['rel_score']] * data['frequency']) 

            if self.score_weights[2] != 0:
 
                if len(all_relation_scores) > 1:
             
                    all_relation_scores = np.array(all_relation_scores)
                    thresholds = threshold_multiotsu(all_relation_scores, classes=5)
                    print("Multi-Otsu thresholds:", thresholds)


                    proportions = []
                    for threshold in thresholds:
                        proportion = np.mean(all_relation_scores >= threshold)
                        proportions.append(proportion)
                    print("Proportions for each threshold:", proportions)

                    cumulative_proportion = 0.0
                    rel_gmm_threshold = thresholds[-1]
                    for i, proportion in enumerate(proportions):
                        cumulative_proportion += proportion
                        if cumulative_proportion > self.score_weights[2]:  
                            rel_gmm_threshold = thresholds[i - 1] if i > 0 else thresholds[0]
                            break

                    self.rel_gmm_threshold = rel_gmm_threshold
                    print("Final rel_gmm_threshold:", self.rel_gmm_threshold)


            token_cache_file = os.path.join(self.output_dir, "token_cache.json")
            os.makedirs(os.path.dirname(token_cache_file), exist_ok=True)
            with open(token_cache_file, "w", encoding="utf-8") as f:
                json.dump(self.token_cache, f, ensure_ascii=False, indent=4)
            print(f"Token cache successfully saved to {token_cache_file}")
            self.data_tr = self.data_tr.map(self.get_rel_score)
            # self.data_ts = self.data_ts.map(self.get_rel_score)
        return self.data_tr, self.data_ts
    
    def get_labels(self):
        self.data_tr = self.data_tr.map(self.filtering_tokens_and_give_labels)
        self.data_ts = self.data_ts.map(self.give_labels)
        return self.data_tr, self.data_ts
    
    def filtering_tokens_and_give_labels(self, example):
        # 解码输入数据
        combined_input_ids = example['input_ids'] 
        out_put_tokens = example['output_tokens']  
        question_length = len(example['input_tokens']) 
   
        perplexity_scores = example.get('ppl_score', [0.0] * len(example['input_ids']))
        attention_scores = example.get('attention_score', [0.0] * len(example['input_ids']))  

        raw_relation_scores = example.get('rel_score', [0.0] * len(example['input_ids'])) 
        relation_scores = [
            score['rel_score'] if isinstance(score, dict) else score
            for score in raw_relation_scores
        ]  
        labels = combined_input_ids .copy() 

        labels[:question_length] = [-100] * (question_length)


        att_threshold = self.value_interest[0] 
        ppl_threshold = self.value_interest[1]  
        rel_threshold = self.value_interest[2] 

        att_percentage = self.percentage_interest[0] 
        ppl_percentage = self.percentage_interest[1]  
        rel_percentage = self.percentage_interest[2]  
        
       
      

        answer_attention_scores = attention_scores[question_length:]  
        answer_perplexity_scores = perplexity_scores[question_length:]  
        answer_relation_scores = relation_scores[question_length:]  
       
        if self.score_weights[0] != 0:
            if len(answer_attention_scores) > 0:
                q1 = np.percentile(answer_attention_scores, 25)  
                q3 = np.percentile(answer_attention_scores, 75) 
                iqr = q3 - q1  
                lower_bound = q1 - 1.5 * iqr  
                upper_bound = q3 + 1.5 * iqr  
            else:
                lower_bound = float('-inf')
                upper_bound = float('inf')
            print(f"Lower bound: {lower_bound}")
           
            if len(answer_attention_scores) > 0:
                below_lower_bound_percentage = np.mean(np.array(answer_attention_scores) < lower_bound) 
                if below_lower_bound_percentage>att_percentage:
                    lower_bound = sorted(answer_attention_scores)[int(len(answer_attention_scores) * att_percentage)]
                below_lower_bound_percentage = np.mean(np.array(answer_attention_scores) < lower_bound) * 100
               
        for i in range(len(out_put_tokens)):
            pos = question_length + i
            if not (
                (self.score_weights[0] == 0 or attention_scores[pos] >= att_threshold and
                lower_bound <= attention_scores[pos]) and
                (self.score_weights[1] == 0 or perplexity_scores[pos] <=0.95 or
                perplexity_scores[pos] <=  sorted(answer_perplexity_scores)[int(len(answer_perplexity_scores) * (1-ppl_percentage))]) and
                (self.score_weights[2] == 0 or relation_scores[pos] >= self.rel_gmm_threshold or
                relation_scores[pos] >= sorted(answer_relation_scores)[int(len(answer_relation_scores) * rel_percentage)])
            ):
                labels[pos] = -100 
        return {'labels': labels}

    def give_labels(self, example):
 
        combined_input_ids = example['input_ids']  
        question_length = len(example['input_tokens'])  

      
        labels = combined_input_ids.copy()  

       
        labels[:question_length] = [-100] * question_length

        return {'labels': labels}

    def get_rel_score(self, example):
        
        input_ids = example['input_ids']
        rel_scores = []  
        with torch.no_grad():
     
            for token_id in input_ids:
                           
              
                token = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
              
               
                rel_scores.append(self.token_cache[token])
                    

               
        return {'rel_score': rel_scores}
        
    
 
    def get_ppl_score(self, example):
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
       
        with torch.no_grad():
       
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            logits = outputs.logits 
            
            
            probs = [0.0]
            for i in range(input_ids.size(1) - 1): 
                current_logits = logits[0, i, :]  
                target_token_id = input_ids[0, i + 1]  
                prob = F.softmax(current_logits, dim=-1)[target_token_id].item() 
                probs.append(prob)

        return {'ppl_score': probs}
    
    def get_attention_score(self, example):
     
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
       
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
            attentions = outputs.attentions

  
        attention_scores = torch.mean(attentions[0], dim=1).squeeze(0)

        attention_scores = torch.mean(attention_scores, dim=1)
 
        attention_scores = attention_scores.squeeze(0).tolist()
     
        return {'attention_score': attention_scores}

    def format_tr_gsm8k(self, example):
        tokenizer = self.tokenizer
        max_length = self.max_length
        question = example['question']
        analysis = example['answer'].split("####")[0].strip()
        answer = f"Answer:\n{example['answer'].split('####')[-1].strip()}"


        
        input_tokens = tokenizer(question, return_tensors='pt')['input_ids']
        output_tokens = tokenizer(analysis, return_tensors='pt', add_special_tokens=False)['input_ids']
        input_ids = torch.cat([input_tokens, output_tokens], dim=1).to(self.device)
        input_ids = input_ids.squeeze(0).tolist()  
        input_tokens = input_tokens.squeeze(0).tolist()
        output_tokens = output_tokens.squeeze(0).tolist()
        
    
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'input_ids': input_ids, 'tokens': tokens, 'question': question, 'answer': answer}
    
    def format_tr_humaneval(self,example):
        tokenizer = self.tokenizer
        max_length = self.max_length
        prompt = example['prompt']
        solution = example['canonical_solution']
        input_tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
        output_tokens = tokenizer(solution, return_tensors='pt', add_special_tokens=False)['input_ids']
    
        input_ids = torch.cat([input_tokens, output_tokens], dim=1).to(self.device)
        input_ids = input_ids.squeeze(0).tolist()
        input_tokens = input_tokens.squeeze(0).tolist()
        output_tokens = output_tokens.squeeze(0).tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
 
        return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'input_ids': input_ids, 'tokens': tokens, 'prompt': prompt, 'solution': solution}

    def format_tr_pubmedqa(self, example):
        tokenizer = self.tokenizer
        instruction = example['instruction']
        input_text = example['input']
        output_text = example['output']

        prompt = f"{instruction}\n{input_text}"
        solution = output_text
        print(f"Prompt: {prompt}")
        print(f"Solution: {solution}")
        input_tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
        output_tokens = tokenizer(solution, return_tensors='pt',add_special_tokens=False)['input_ids']

        input_ids = torch.cat([input_tokens, output_tokens], dim=1).to(self.device)
        input_ids = input_ids.squeeze(0).tolist()
        input_tokens = input_tokens.squeeze(0).tolist()
        output_tokens = output_tokens.squeeze(0).tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'input_ids': input_ids, 'tokens': tokens, 'instruction': instruction, 'input': input_text, 'output': output_text}

    def format_tr_fiqa(self, example):
        tokenizer = self.tokenizer
        instruction = example['instruction']
        input_text = example['input']
        output_text = example['output']

        prompt = f"{instruction}\n{input_text}"
        solution = output_text
        input_tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
        output_tokens = tokenizer(solution, return_tensors='pt',add_special_tokens=False)['input_ids']

        input_ids = torch.cat([input_tokens, output_tokens], dim=1).to(self.device)
        input_ids = input_ids.squeeze(0).tolist()
        input_tokens = input_tokens.squeeze(0).tolist()
        output_tokens = output_tokens.squeeze(0).tolist()

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'input_ids': input_ids, 'tokens': tokens, 'instruction': instruction, 'input': input_text, 'output': output_text}

    def format_tr_codeexercise(self, example):
    
        tokenizer = self.tokenizer
        max_length = self.max_length

 
        human_content = None
        bot_content = None
     
        if 'chat_rounds' not in example:
            print(f"Invalid example format: {example}")
            return None

     
        for entry in example['chat_rounds']:
            if entry['role'] == 'human':
                human_content = entry['content']
            elif entry['role'] == 'bot':
                bot_content = entry['content']


        if not human_content or not bot_content:
            return None


        prompt = f"{human_content.strip()}"
        solution = f"{bot_content.strip()}"
        print(f"Prompt: {prompt}")
        print(f"Solution: {solution}")

        input_tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
        output_tokens = tokenizer(solution, return_tensors='pt',add_special_tokens=False)['input_ids']


        output_tokens = output_tokens[:, 1:]


        input_ids = torch.cat([input_tokens, output_tokens], dim=1).to(self.device)
        input_ids = input_ids.squeeze(0).tolist() 
        input_tokens = input_tokens.squeeze(0).tolist()
        output_tokens = output_tokens.squeeze(0).tolist()

 
        tokens = tokenizer.convert_ids_to_tokens(input_ids)


        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_ids': input_ids,
            'tokens': tokens,
            'prompt': prompt,
            'solution': solution
        }

    def format_tr_numinamathcot(self, example):
        tokenizer = self.tokenizer
        max_length = self.max_length
        question = example['problem']
        answer = example['solution']

        input_tokens = tokenizer(question, return_tensors='pt')['input_ids']
        output_tokens = tokenizer(answer, return_tensors='pt', add_special_tokens=False)['input_ids']
        input_ids = torch.cat([input_tokens, output_tokens], dim=1).to(self.device)
        input_ids = input_ids.squeeze(0).tolist()  
        input_tokens = input_tokens.squeeze(0).tolist()
        output_tokens = output_tokens.squeeze(0).tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'input_ids': input_ids, 'tokens': tokens, 'question': question, 'answer': answer}

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name,trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    print("Loading dataset...")
    
    if args.dataset_name == "modelscope/gsm8k":
        data_tr = MsDataset.load(args.dataset_name, subset_name='main', split='train')
        data_ts = MsDataset.load(args.dataset_name, subset_name='main', split='test').select(range(1))
    elif args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        data_tr = MsDataset.load(args.dataset_name, subset_name='default', split='train')
        data_ts = MsDataset.load('modelscope/humaneval', subset_name='openai_humaneval', split='test').select(range(1))
    elif args.dataset_name == "modelscope/humaneval":
        data_tr = MsDataset.load(args.dataset_name, subset_name='default', split='train')
        data_ts = MsDataset.load(args.dataset_name, subset_name='default', split='train').select(range(1))
    elif args.dataset_name == "hiyouga/PubMedQA":
        data_tr = MsDataset.load(args.dataset_name, subset_name='default', split='train')
        data_ts = MsDataset.load(args.dataset_name, subset_name='default', split='test').select(range(1))
    elif args.dataset_name == "hiyouga/fiqa":
        data_tr = MsDataset.load(args.dataset_name, subset_name='default', split='train')
        data_ts = MsDataset.load(args.dataset_name, subset_name='default', split='test').select(range(1))
    elif args.dataset_name == "AI-MO/NuminaMath-CoT":
        data_tr = MsDataset.load(args.dataset_name, subset_name='default', split='train', trust_remote_code=True).select(range(3000))
        data_ts = MsDataset.load(args.dataset_name, subset_name='default', split='test', trust_remote_code=True).select(range(1))
    

    if args.dataset_name == "modelscope/gsm8k":
        domain_words = ["1","2","3","4","5","6","7","8","9","10","+","-","*","/","=","maths"]
    elif args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        domain_words = ["def", "return", "if", "else", "for", "while", "import", "from", "class", "try"]
    elif args.dataset_name == "modelscope/humaneval":
        domain_words = ["def", "return", "if", "else", "for", "while", "import", "from", "class", "try"]
    elif args.dataset_name == "hiyouga/PubMedQA":
        domain_words = [
                "gene", "protein", "cell", "study", "research", "data", "result", "method", 
                "analysis", "expression", "treatment", "disease", "patient", "clinical", 
                "trial", "therapy", "diagnosis", "symptom", "risk", "factor", "medicine", 
                "health", "biomarker", "mutation", "pathway", "drug", "effect", "outcome", 
                "sample", "control", "group", "placebo", "dose", "response", "mechanism", 
                "inflammation", "immune", "system", "cancer", "tumor", "infection", 
                "bacteria", "virus", "antibody", "vaccine", "genome", "sequencing", 
                "epidemiology", "prevalence", "incidence", "mortality", "survival", 
                "biological", "pathology", "biochemistry", "pharmacology", "neuroscience", 
                "cardiology", "oncology", "radiology", "surgery", "psychiatry", "dermatology"
            ]
    elif args.dataset_name == "hiyouga/fiqa":
        domain_words = [
                "stock", "market", "investment", "finance", "economy", "trading", 
                "shares", "portfolio", "dividend", "revenue", "profit", "loss", 
                "capital", "equity", "bond", "interest", "rate", "loan", "bank", 
                "credit", "debt", "fund", "valuation", "asset", "liability", 
                "cash", "earnings", "forecast", "growth", "risk", "return", 
                "inflation", "tax", "expense", "income", "budget", "accounting"
            ]    
    elif args.dataset_name == "AI-MO/NuminaMath-CoT":
        domain_words = ["1","2","3","4","5","6","7","8","9","10","+","-","*","/","=","maths"]




    output_dir = os.path.dirname(args.output_data_file) if os.path.dirname(args.output_data_file) else "data"
    dataset_editor = DatasetEditor(data_tr, data_ts, model, tokenizer, None, domain_words,
                                   score_weights=args.score_weights, value_interest=args.value_interest,
                                   percentage_interest=args.percentage_interest, max_length=args.max_length, device=device,
                                   output_dir=output_dir)
    

    if args.dataset_name == "modelscope/gsm8k":
        dataset_editor.edit_data_gsm8k()
    elif args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        dataset_editor.edit_data_codeexercise()
    elif args.dataset_name == "modelscope/humaneval":
        dataset_editor.edit_data_humaneval()
    elif args.dataset_name == "hiyouga/PubMedQA":
        dataset_editor.edit_data_pubmedqa()
    elif args.dataset_name == "hiyouga/fiqa":
        dataset_editor.edit_data_fiqa()
    elif args.dataset_name == "AI-MO/NuminaMath-CoT":
        dataset_editor.edit_data_numinamathcot()


    dataset_editor.get_score()
    data_tr, data_ts = dataset_editor.get_labels()

    output_data = []
    output_test_data = []
    token_scores = []

    for example in data_tr:
        question_length = len(example['input_tokens'])  
        tokens = example['tokens']
        attention_scores = example.get('attention_score', [0.0] * len(example['input_ids']))  
        perplexity_scores = example.get('ppl_score', [0.0] * len(example['input_ids'])) 
        relation_scores = example.get('rel_score', [0.0] * len(example['input_ids']))  
     
        for i in range(len(tokens) - question_length):
            pos = question_length + i
            token_scores.append({
                'token': tokens[pos],
                'attention_score': attention_scores[pos],
                'perplexity_score': perplexity_scores[pos],
                'relation_score': relation_scores[pos]
            })

    
        output_data.append({
            "input_ids": example['input_ids'],
            "labels": example['labels']
        })
    
    for example in data_ts:
         output_test_data.append({
            "input_ids": example['input_ids'],
            "labels": example['labels']
        })

    output_data_file = args.output_data_file


    os.makedirs(os.path.dirname(output_data_file), exist_ok=True)

    with open(output_data_file, "w", encoding="utf-8") as f:
        for example in output_data:
            f.write(json.dumps(example) + '\n')
    
    output_dir = os.path.dirname(args.output_data_file) if os.path.dirname(args.output_data_file) else "data"
    token_scores_file = os.path.join(output_dir, "token_scores.json")

    os.makedirs(os.path.dirname(token_scores_file), exist_ok=True)

    token_scores = []
    for example in data_tr:
        question_length = len(example['input_tokens']) 
        tokens = example['tokens']
        attention_scores = example.get('attention_score', [0.0] * len(example['input_ids'])) 
        perplexity_scores = example.get('ppl_score', [0.0] * len(example['input_ids'])) 
        relation_scores = example.get('rel_score', [0.0] * len(example['input_ids']))  


        for i in range(len(tokens) - question_length):
            pos = question_length + i
            token_scores.append({
                'token': tokens[pos],
                'attention_score': attention_scores[pos],
                'perplexity_score': perplexity_scores[pos],
                'relation_score': relation_scores[pos]
            })

  
    with open(token_scores_file, "w", encoding="utf-8") as f:
        json.dump(token_scores, f, ensure_ascii=False, indent=4)

 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTA Command Line Script")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="modelscope/gsm8k", help="Dataset name")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    parser.add_argument("--score_weights", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Score weights [att, ppl, rel]")
    parser.add_argument("--value_interest", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Value interest thresholds")
    parser.add_argument("--percentage_interest", type=float, nargs=3, default=[0.1, 0.1, 0.1], help="Percentage interest thresholds")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--output_data_file", type=str, default="data/augmented_gsm8k.txt", help="Path to save the output data")  

    args = parser.parse_args()

    main(args)
    