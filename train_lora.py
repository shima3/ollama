import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import load_dataset
import bitsandbytes as bnb
from datetime import datetime
import stat

def create_modelfile(base_model_name: str, gguf_path: str, tokenizer: AutoTokenizer):
    """
    指定されたGGUFファイル用のModelfileを自動生成し、提案されるOllamaタグを返す。
    """
    gguf_filename = os.path.basename(gguf_path)
    modelfile_path = os.path.join(os.path.dirname(gguf_path), "Modelfile")

    template = ""
    parameters = []
    # (テンプレート判定ロジックは変更なし)
    model_name_lower = base_model_name.lower()
    if tokenizer.chat_template:
        # (省略)
        if "[INST] <<SYS>>" in tokenizer.chat_template:
            template = 'TEMPLATE """[INST] <<SYS>>\n{{ .System }}\n<</SYS>>\n\n{{ .Prompt }} [/INST]"""'
            parameters.extend(['PARAMETER stop "[INST]"', 'PARAMETER stop "[/INST]"'])
        elif "[INST] {{ .Prompt }} [/INST]" in tokenizer.chat_template or "<s>[INST] {{ .Prompt }} [/INST]" in tokenizer.chat_template:
            template = 'TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""'
            parameters.extend(['PARAMETER stop "[INST]"', 'PARAMETER stop "[/INST]"'])
    if not template:
        # (省略)
        if "llama-2" in model_name_lower and "instruct" in model_name_lower:
            template = 'TEMPLATE """[INST] <<SYS>>\n{{ .System }}\n<</SYS>>\n\n{{ .Prompt }} [/INST]"""'
            parameters.extend(['PARAMETER stop "[INST]"', 'PARAMETER stop "[/INST]"'])

    content = f"FROM ./{gguf_filename}\n"
    if template: content += f"{template.strip()}\n"
    if parameters: content += "\n".join(parameters) + "\n"

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ Successfully created Modelfile at: {modelfile_path}")
    
    # --- 変更点: モデルタグを ":latest" に固定 ---
    model_tag_base = os.path.basename(os.path.dirname(modelfile_path))
    suggested_model_tag = f"{model_tag_base}:latest"
    
    return suggested_model_tag, modelfile_path

def generate_run_artifacts(output_dir: str, args: argparse.Namespace, suggested_model_tag: str, modelfile_path: str):
    """
    トレーニング情報ファイルとOllama作成スクリプトを生成する。
    """
    # 1. training_info.txt の作成
    info_path = os.path.join(output_dir, "training_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*40 + "\n")
        f.write("Parameters:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("="*40 + "\n")
    print(f"✅ Successfully created training info file at: {info_path}")

    # 2. create_ollama_model.sh の作成
    script_path = os.path.join(output_dir, "create_ollama_model.sh")
    script_content = f"""#!/bin/sh
cd "$(dirname "$0")"
ollama create {suggested_model_tag} -f {os.path.basename(modelfile_path)}
echo "✅ Ollama model '{suggested_model_tag}' created successfully."
"""
    with open(script_path, "w", encoding="utf-8", newline='\n') as f:
        f.write(script_content)
    
    os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    print(f"✅ Successfully created executable script at: {script_path}")
    print(f"\nTo create the Ollama model, run the following command:")
    print("="*70)
    print(f"cd {output_dir} && ./create_ollama_model.sh")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA, convert to GGUF, and create run artifacts.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("base_model", type=str, help="Base model name or path.")
    parser.add_argument("dataset", type=str, help="Dataset JSON file path.")
    parser.add_argument("output_dir", type=str, help="Directory to save the final GGUF model and all run artifacts.")
    parser.add_argument("--target-modules", type=str, required=True, help="Comma-separated list of target modules for LoRA.")
    parser.add_argument("-q", "--quantization", type=int, choices=[4, 8, 16], default=4, help="Quantization bits for the final GGUF file. Default: 4.")
    parser.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank. Default: 16.")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    parser.add_argument("-e", "--epoch", type=int, default=5, help="Number of training epochs. Default: 5.")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    output_base_name = os.path.basename(os.path.normpath(args.output_dir))
    
    # --- 変更点: ファイル名から量子化情報を削除 ---
    final_gguf_filename = f"{output_base_name}.gguf"
    final_gguf_path = os.path.join(args.output_dir, final_gguf_filename)
    
    # (ファインチューニングのプロセスは変更なし)
    # ... (省略) ...
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    dataset = load_dataset("json", data_files=args.dataset)["train"]
    def preprocess_function(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=2048)
    tokenized_dataset = dataset.map(preprocess_function, remove_columns=list(dataset.features))
    lora_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=[m.strip() for m in args.target_modules.split(',')], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    training_output_dir = tempfile.mkdtemp()
    training_args = TrainingArguments(output_dir=training_output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4, num_train_epochs=args.epoch, logging_steps=10, fp16=True, save_strategy="epoch", report_to="none")
    trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print("\nStarting fine-tuning process..."); trainer.train(); print("Fine-tuning completed.")
    checkpoint_dirs = [d for d in os.listdir(training_output_dir) if d.startswith("checkpoint-")]
    if not checkpoint_dirs: raise RuntimeError("No checkpoint found after training.")
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda d: int(d.split('-')[1]))[-1]
    adapter_path = os.path.join(training_output_dir, latest_checkpoint)
    del model, peft_model, trainer; torch.cuda.empty_cache()
    print("\nReloading base model in FP16 for clean merging...")
    base_model_fp16 = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    print(f"Loading adapter from {adapter_path}..."); merged_model = PeftModel.from_pretrained(base_model_fp16, adapter_path)
    print("Merging adapter into FP16 model..."); merged_model = merged_model.merge_and_unload()
    
    print("\nStarting GGUF conversion process...")
    merged_model_dir = tempfile.mkdtemp()
    fp16_gguf_path = ""
    try:
        merged_model.save_pretrained(merged_model_dir); tokenizer.save_pretrained(merged_model_dir)
        llama_cpp_path = os.environ.get("LLAMA_CPP_PATH");
        if not llama_cpp_path: raise ValueError("LLAMA_CPP_PATH environment variable not set.")
        convert_script_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script_path): raise FileNotFoundError(f"'convert_hf_to_gguf.py' not found in {llama_cpp_path}")
        
        # --- 変更点: 一時的なFP16ファイルパスの定義 ---
        fp16_gguf_path = os.path.join(args.output_dir, f"{output_base_name}-fp16.gguf")
        
        print(f"\nStep 1: Running GGUF conversion (to FP16)...")
        subprocess.run([sys.executable, convert_script_path, merged_model_dir, "--outfile", fp16_gguf_path, "--outtype", "f16"], check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)

        if args.quantization == 16:
            # FP16が最終目標の場合、一時ファイルをリネームする
            print("Renaming FP16 GGUF to the final filename...")
            shutil.move(fp16_gguf_path, final_gguf_path)
        else:
            # 量子化する場合
            quantize_exec_name = "llama-quantize"
            quantize_exec_path = os.path.join(llama_cpp_path, "build", "bin", quantize_exec_name)
            if not os.path.exists(quantize_exec_path): quantize_exec_path = os.path.join(llama_cpp_path, quantize_exec_name)
            if not os.path.exists(quantize_exec_path): raise FileNotFoundError(f"'{quantize_exec_name}' not found. Please build llama.cpp.")
            
            quant_map_cli = {8: "Q8_0", 4: "Q4_K_M"}
            quant_type_cli = quant_map_cli[args.quantization]
            
            print(f"\nStep 2: Running GGUF quantization (to {quant_type_cli})...")
            subprocess.run([quantize_exec_path, fp16_gguf_path, final_gguf_path, quant_type_cli], check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
            # 一時的なFP16ファイルを削除
            os.remove(fp16_gguf_path)
        
        print(f"\n✅ Successfully created GGUF model at: {final_gguf_path}")
        
        suggested_model_tag, modelfile_path = create_modelfile(args.base_model, final_gguf_path, tokenizer)
        generate_run_artifacts(args.output_dir, args, suggested_model_tag, modelfile_path)

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
    finally:
        print("\nCleaning up temporary directories...")
        shutil.rmtree(training_output_dir)
        shutil.rmtree(merged_model_dir)

if __name__ == "__main__":
    main()
