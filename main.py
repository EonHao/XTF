import argparse
import os

def run_xtf_script(model_name, dataset_name, score_weights, percentage_interest, output_data_file,gpu):
    print("Running XTF script...")
    os.system(f"python XTF_script.py --model_name {model_name} --dataset_name {dataset_name} "
              f"--score_weights {' '.join(map(str, score_weights))} --percentage_interest {' '.join(map(str, percentage_interest))} "
              f"--output_data_file {output_data_file}  --gpu {gpu}")

def run_loraft_script(data_path, model_name, adapter_save_path, batch_size, epochs, learning_rate, lora_r, lora_alpha, lora_dropout, target_modules, dataset_name, gpu):
    print("Running LoRaFT script...")
    os.system(f"python LoRaFT_script.py --data_path {data_path} --model_name {model_name} "
              f"--adapter_save_path {adapter_save_path} --batch_size {batch_size} --epochs {epochs} "
              f"--learning_rate {learning_rate} --lora_r {lora_r} --lora_alpha {lora_alpha} "
              f"--lora_dropout {lora_dropout} --dataset_name {dataset_name} --gpu {gpu}")

def run_sft_script(data_path,  model_name, batch_size, epochs, learning_rate, save_path,dataset_name, gpu):
    print("Running SFT script...")
    os.system(f"python SFT_script.py --data_path {data_path} --model_name {model_name} "
              f"--batch_size {batch_size} --epochs {epochs} --learning_rate {learning_rate} "
              f"--save_path {save_path} --dataset_name {dataset_name} --gpu {gpu}")

def run_eval_script(original_model_path, model1_path, model2_path, dataset_name,  batch_size,  use_lora, k_values, gpu):
    print("Running Eval script...")
    os.system(f"python Eval_script.py --original_model_path {original_model_path} --model1_path {model1_path} "
              f"--model2_path {model2_path} --dataset_name {dataset_name}  "
              f"--batch_size {batch_size}  --use_lora {use_lora} --k_values {' '.join(map(str, k_values))} --gpu {gpu}")

def main():
    parser = argparse.ArgumentParser(description="main script for model training and evaluation")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="model name")
    parser.add_argument("--train_dataset_name", type=str, default="modelscope/gsm8k", help="training dataset name")
    parser.add_argument("--score_weights", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Not 0 if you want to use the corresponding score")
    parser.add_argument("--percentage_interest", type=float, nargs=3, default=[0.05, 0.1, 0.05], help="Percentage of interest for each score")
    parser.add_argument("--output_data_file1", type=str, default="data/augmented_gsm8k.txt", help="The path to the original output data file")
    parser.add_argument("--output_data_file2", type=str, default="data/augmented_gsm8k.txt", help="The path to the augmented output data file")
    parser.add_argument("--save_model_path1", type=str, default="lora_adapters1", help="model path 1")
    parser.add_argument("--save_model_path2", type=str, default="lora_adapters2", help="model path 2")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", type=str, nargs="+", default=None)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1])
    parser.add_argument("--use_lora", type=int, default=1)
    args = parser.parse_args()


    run_xtf_script(
        model_name=args.model_name,
        score_weights=[0.0,0.0,0.0],
        percentage_interest=[0.0,0.0,0.0],
        dataset_name=args.train_dataset_name,
        output_data_file=args.output_data_file1,
        gpu=args.gpu
    )

   
    if args.use_lora == 1:
        run_loraft_script(
            data_path=args.output_data_file1,
            model_name=args.model_name,
            adapter_save_path=args.save_model_path1,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            dataset_name=args.train_dataset_name,
            gpu=args.gpu
        )
    else:
        run_sft_script(
            data_path=args.output_data_file1,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            dataset_name=args.train_dataset_name,
            save_path=args.save_model_path1,
            gpu=args.gpu
        )


    run_xtf_script(
        model_name=args.model_name,
        dataset_name=args.train_dataset_name,
        score_weights=args.score_weights,
        output_data_file=args.output_data_file2,
        percentage_interest=args.percentage_interest,
        gpu=args.gpu
    )


    if args.use_lora == 1:
        run_loraft_script(
            data_path=args.output_data_file2,
            model_name=args.model_name,
            adapter_save_path=args.save_model_path2,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            dataset_name=args.train_dataset_name,
            gpu=args.gpu
        )
    else:
        run_sft_script(
            data_path=args.output_data_file2,
            model_name=args.model_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            save_path=args.save_model_path2,         
            dataset_name=args.train_dataset_name,   
            gpu=args.gpu
        )



    run_eval_script(
        original_model_path=args.model_name,
        model1_path=args.save_model_path1,
        model2_path=args.save_model_path2,
        dataset_name=args.train_dataset_name,
        batch_size=args.batch_size*2,
        use_lora=args.use_lora,
        k_values=args.k_values,
        gpu=args.gpu
    )

if __name__ == "__main__":
    main()
    