import argparse
import os
import json
import torch
import random
import numpy as np
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.msdatasets import MsDataset
import utils.dataset_eval as dataset_eval

def load_original_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
     
    tokenizer.padding_side = 'left'
 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    return model, tokenizer


def load_lora_model(base_model_path, lora_path, device):
    from peft import LoraConfig, get_peft_model, PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True).to(device)
    lora_model = PeftModel.from_pretrained(base_model, lora_path).to(device)
    return lora_model

def load_finetuned_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on specific datasets.")
    parser.add_argument("--original_model_path", type=str, required=True, help="Path to the original model.")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to the first fine-tuned model.")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to the second fine-tuned model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name.")
    parser.add_argument("--subset_name", type=str, default="main", help="Subset name of the dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--test_data_size", type=int, default=200, help="Number of test samples to use.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument("--use_lora", type=int, default=0, help="Whether to use LoRA models.")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1], help="k values for HumanEval evaluation.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use.")
    parser.add_argument("--judge_api_url", type=str, default=None, help="API URL for judge model evaluation.")
    parser.add_argument("--judge_api_key", type=str, default=None, help="API key for judge model evaluation.")
    parser.add_argument("--eval_output_file", type=str, default="evaluation_results.txt", help="Output file for evaluation results.")
    parser.add_argument("--math_log_file", type=str, default="math500_eval.txt", help="Log file for math evaluation.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == "modelscope/gsm8k":
        data = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test")
    elif args.dataset_name == "modelscope/humaneval" or args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        data = MsDataset.load("modelscope/humaneval", subset_name="openai_humaneval", split="test")
    elif args.dataset_name == "hiyouga/PubMedQA":
        data = MsDataset.load("hiyouga/PubMedQA", subset_name="default", split="test")
    elif args.dataset_name == "hiyouga/fiqa":
        data = MsDataset.load("hiyouga/fiqa", subset_name="default", split="test")
    elif args.dataset_name == "AI-MO/NuminaMath-CoT":
        data = MsDataset.load("AI-ModelScope/MATH-500", subset_name="default", split="test")
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")



    original_model, tokenizer = load_original_model(args.original_model_path, device)
    if args.dataset_name == "modelscope/gsm8k":
        original_model_accuracy = dataset_eval.evaluate_gsm8k(data, original_model, tokenizer, batch_size=args.batch_size, output_file=args.eval_output_file)
        print(f"Accuracy on original model: {original_model_accuracy:.4f}")
    elif args.dataset_name == "modelscope/humaneval" or args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        original_model_pass_k = dataset_eval.evaluate_humaneval(data, original_model, tokenizer, k_values=args.k_values)
        print(f"pass@k on original model: {original_model_pass_k}")
    elif args.dataset_name == "hiyouga/PubMedQA":
        original_model_bleu = dataset_eval.evaluate_pubmedqa_judge_model(
            data, original_model, tokenizer, batch_size=args.batch_size,
            api_url=args.judge_api_url, api_key=args.judge_api_key
        )
        print(f"Accuracy on original model: {original_model_bleu:.4f}")
    elif args.dataset_name == "hiyouga/fiqa":
        original_model_bleu = dataset_eval.evaluate_fiqa_judge_model(
            data, original_model, tokenizer, batch_size=args.batch_size,
            api_url=args.judge_api_url, api_key=args.judge_api_key
        )
        print(f"Judge model score on original model: {original_model_bleu:.4f}")
    elif args.dataset_name == "AI-MO/NuminaMath-CoT":
        original_model_accuracy = dataset_eval.evaluate_math500(data, original_model, tokenizer, batch_size=args.batch_size, log_file=args.math_log_file)
        print(f"Accuracy on original model: {original_model_accuracy:.4f}")
    del original_model  
    torch.cuda.empty_cache()

   
  
    if args.use_lora == 1:
        model1 = load_lora_model(args.original_model_path, args.model1_path, device)
    else:
        model1 = load_finetuned_model(args.model1_path, device)
    if args.dataset_name == "modelscope/gsm8k":
        model1_accuracy = dataset_eval.evaluate_gsm8k(data, model1, tokenizer, batch_size=args.batch_size, output_file=args.eval_output_file)
        print(f"Accuracy on fine tuned model1: {model1_accuracy:.4f}")
    elif args.dataset_name == "modelscope/humaneval" or args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        model1_pass_k = dataset_eval.evaluate_humaneval(data, model1, tokenizer, k_values=args.k_values)
        print(f"pass@k on fine tuned model1: {model1_pass_k}")
    elif args.dataset_name == "hiyouga/PubMedQA":
        model1_bleu = dataset_eval.evaluate_pubmedqa_judge_model(
            data, model1, tokenizer, batch_size=args.batch_size,
            api_url=args.judge_api_url, api_key=args.judge_api_key
        )
        print(f"Accuracy on fine tuned model1: {model1_bleu:.4f}")
    elif args.dataset_name == "hiyouga/fiqa":
        model1_bleu = dataset_eval.evaluate_fiqa_judge_model(
            data, model1, tokenizer, batch_size=args.batch_size,
            api_url=args.judge_api_url, api_key=args.judge_api_key
        )
        print(f"Judge model score on fine tuned model1: {model1_bleu:.4f}")
    elif args.dataset_name == "AI-MO/NuminaMath-CoT":
        model1_accuracy = dataset_eval.evaluate_math500(data, model1, tokenizer, batch_size=args.batch_size, log_file=args.math_log_file)
        print(f"Accuracy on fine tuned model1: {model1_accuracy:.4f}")
    del model1 
    torch.cuda.empty_cache()

    
  
    if args.use_lora == 1:
        model2 = load_lora_model(args.original_model_path, args.model2_path, device)
    else:
        model2 = load_finetuned_model(args.model2_path, device)
    if args.dataset_name == "modelscope/gsm8k":
        model2_accuracy = dataset_eval.evaluate_gsm8k(data, model2, tokenizer, batch_size=args.batch_size, output_file=args.eval_output_file)
        print(f"Accuracy on fine tuned model2: {model2_accuracy:.4f}")
    elif args.dataset_name == "modelscope/humaneval" or args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        model2_pass_k = dataset_eval.evaluate_humaneval(data, model2, tokenizer, k_values=args.k_values)
        print(f"pass@k on fine tuned model2: {model2_pass_k}")
    elif args.dataset_name == "hiyouga/PubMedQA":
        model2_bleu = dataset_eval.evaluate_pubmedqa_judge_model(
            data, model2, tokenizer, batch_size=args.batch_size,
            api_url=args.judge_api_url, api_key=args.judge_api_key
        )
        print(f"Accuracy on fine tuned model2: {model2_bleu:.4f}")
    elif args.dataset_name == "hiyouga/fiqa":
        model2_bleu = dataset_eval.evaluate_fiqa_judge_model(
            data, model2, tokenizer, batch_size=args.batch_size,
            api_url=args.judge_api_url, api_key=args.judge_api_key
        )
        print(f"Judge model score on fine tuned model2: {model2_bleu:.4f}")
    elif args.dataset_name == "AI-MO/NuminaMath-CoT":
        model2_accuracy = dataset_eval.evaluate_math500(data, model2, tokenizer, batch_size=args.batch_size, log_file=args.math_log_file)
        print(f"Accuracy on fine tuned model2: {model2_accuracy:.4f}")
    del model2  
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()