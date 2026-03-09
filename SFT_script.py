import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
import json
import time
import random
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.msdatasets import MsDataset
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
                    line = line.replace("'", '"')
                    item = json.loads(line)

                    labels = item["labels"]
                    if isinstance(labels, int):
                        labels = [labels]

                    input_ids = item["input_ids"]
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

def train(model, train_dataloader, epochs, lr, device, save_path, tokenizer):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_accuracy = 0  

    for epoch in range(epochs):
        start_time = time.time() 
        total_loss = 0
        for batch in train_dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            torch.cuda.empty_cache()
     
        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_dataloader):.4f}")
        
        end_time = time.time() 
        epoch_time = end_time - start_time  
        print(f"Epoch {epoch+1}, Training Time: {epoch_time:.2f} seconds") 
       
    model.save_pretrained(save_path)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the fine-tuned model.")
    parser.add_argument("--gpu", type=str, required=True, help="GPU device ID to use.")  
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    train_dataset = TextDataset(args.data_path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=8,
        pin_memory=True
    )


    train(model, train_dataloader, args.epochs, args.learning_rate, device, args.save_path, tokenizer)

if __name__ == "__main__":
    main()