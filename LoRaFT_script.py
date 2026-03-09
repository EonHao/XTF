import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
import random 
from modelscope.msdatasets import MsDataset
from modelscope import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from utils.dataset_eval import evaluate_gsm8k, evaluate_humaneval, evaluate_pubmedqa_judge_model, evaluate_fiqa_judge_model


def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    labels_padded = pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded
    }


class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    input_ids = item["input_ids"]
                    labels = item["labels"]

                    if len(labels) == 1:
                        labels = [-100] * (len(input_ids) - 1) + labels
                    elif len(labels) != len(input_ids):
                        continue

                    self.data.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long)
                    })
                except:
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(model, train_dataloader, eval_data, eval_function, epochs, lr, device, adapter_save_path, tokenizer):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

   

    for epoch in range(epochs):
        start_time = time.time()  
        total_loss = 0
        for batch in train_dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        end_time = time.time()  
        epoch_time = end_time - start_time  
        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_dataloader):.4f}")
        print(f"Epoch {epoch+1}, Training Time: {epoch_time:.2f} seconds")  # Print training time
        accuracy = eval_function(eval_data, model, tokenizer)
        print(f"Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}")

    model.save_pretrained(adapter_save_path)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model with LoRA.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--adapter_save_path", type=str, required=True, help="Path to save the LoRA adapter.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA parameter r.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA parameter alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout rate.")
    parser.add_argument("--target_modules", type=str, nargs="+", default=None, help="Target modules for LoRA.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use.")
    parser.add_argument("--dataset_name", type=str, required=False, help="Dataset name for evaluation.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextDataset(args.data_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

 
    if args.dataset_name == "modelscope/gsm8k":
        dataset = MsDataset.load("modelscope/gsm8k", subset_name="main", split="test")
        eval_data = dataset.select(random.sample(range(len(dataset)), 100))  # Randomly select 100 samples
    elif args.dataset_name == "modelscope/humaneval" or args.dataset_name == "codefuse-ai/CodeExercise-Python-27k":
        eval_function = evaluate_humaneval
        dataset = MsDataset.load("modelscope/humaneval", subset_name="openai_humaneval", split="test")
        eval_data = dataset.select(random.sample(range(len(dataset)), 100))  # Randomly select 100 samples
    elif args.dataset_name == "hiyouga/PubMedQA":
        eval_function = evaluate_pubmedqa_judge_model
        dataset = MsDataset.load("hiyouga/PubMedQA", subset_name="default", split="test")
        eval_data = dataset.select(random.sample(range(len(dataset)), 100))  # Randomly select 100 samples  
    elif args.dataset_name == "hiyouga/fiqa":
        eval_function = evaluate_fiqa_judge_model
        dataset = MsDataset.load("hiyouga/fiqa", subset_name="default", split="test")
        eval_data = dataset.select(random.sample(range(len(dataset)), 100))  # Randomly select 100 samples
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    trainable_model = get_peft_model(model, lora_config)

    train(trainable_model, train_dataloader, eval_data, eval_function, args.epochs, args.learning_rate, device, args.adapter_save_path, tokenizer)

if __name__ == "__main__":
    main()