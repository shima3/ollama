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

def create_modelfile(base_model_name: str, gguf_path: str):
    """
    指定されたGGUFファイル用のModelfileを自動生成する。
    """
    gguf_filename = os.path.basename(gguf_path)
    modelfile_path = os.path.join(os.path.dirname(gguf_path) or ".", "Modelfile")

    template = ""
    parameters = []

    model_name_lower = base_model_name.lower()
    # モデル名に基づいてチャットテンプレートを自動設定
    if "llama-2" in model_name_lower and "instruct" in model_name_lower:
        print("Detected Llama-2-Instruct style model. Generating appropriate Modelfile.")
        template = '''
TEMPLATE """[INST] <<SYS>>
{{ .System }}
<</SYS>>

{{ .Prompt }} [/INST]"""
'''
        parameters.append('PARAMETER stop "[INST]"')
        parameters.append('PARAMETER stop "[/INST]"')
    elif "codellama" in model_name_lower and "instruct" in model_name_lower:
        print("Detected CodeLlama-Instruct style model. Generating appropriate Modelfile.")
        template = '''
TEMPLATE """[INST] {{ .System }} {{ .Prompt }} [/INST]"""
'''
        parameters.append('PARAMETER stop "[INST]"')
        parameters.append('PARAMETER stop "[/INST]"')
    else:
        print(f"Warning: Could not determine a specific chat template for '{base_model_name}'.")
        print("A generic Modelfile will be created. You may need to edit it manually.")

    content = f"FROM ./{gguf_filename}\n"
    if template: content += f"{template.strip()}\n"
    if parameters: content += "\n".join(parameters) + "\n"

    try:
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"\n✅ Successfully created Modelfile at: {modelfile_path}")
        print("\nTo create the model in Ollama, run the following command in your terminal:")
        
        suggested_model_name = os.path.splitext(gguf_filename)[0]
        print("="*70)
        print(f"ollama create {suggested_model_name} -f {modelfile_path}")
        print("="*70)

    except Exception as e:
        print(f"\nCould not create Modelfile: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA and convert to GGUF.", formatter_class=argparse.RawTextHelpFormatter)
    
    # 必須引数を明確化
    parser.add_argument("base_model", type=str, help="Base model name or path (e.g., 'meta-llama/Llama-2-7b-chat-hf').")
    parser.add_argument("dataset", type=str, help="Dataset JSON file path.")
    parser.add_argument("output_model", type=str, help="Final GGUF model file path (e.g., './my-model.gguf').")
    
    # オプション引数
    parser.add_argument("--target-modules", type=str, required=True, help="Comma-separated list of target modules for LoRA (e.g., 'q_proj,v_proj'). Use find_modules.py to discover these.")
    parser.add_argument("-q", "--quantization", type=int, choices=[4, 8, 16], default=4, help="Quantization bits for the final GGUF file. Default: 4.")
    parser.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank. Default: 16.")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    parser.add_argument("-e", "--epoch", type=int, default=5, help="Number of training epochs. Default: 5.")
    
    args = parser.parse_args()
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    
    def preprocess_function(example):
        # Hugging Faceの標準的なチャットテンプレート形式を適用
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=2048, padding=False)
    
    dataset = load_dataset("json", data_files=args.dataset)["train"]
    tokenized_dataset = dataset.map(preprocess_function, remove_columns=list(dataset.features))
    
    lora_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=[m.strip() for m in args.target_modules.split(',')], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    output_dir = tempfile.mkdtemp()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=args.epoch,
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        report_to="none"
    )
    trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    
    print("\nStarting fine-tuning process...")
    trainer.train()
    print("Fine-tuning completed.")
    
    # 最新のチェックポイントからアダプタをロード
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoint_dirs: raise RuntimeError("No checkpoint found after training.")
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda d: int(d.split('-')[1]))[-1]
    adapter_path = os.path.join(output_dir, latest_checkpoint)
    
    del model, peft_model, trainer
    torch.cuda.empty_cache()

    print("\nReloading base model in FP16 for clean merging...")
    base_model_fp16 = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    
    print(f"Loading adapter from {adapter_path}...")
    merged_model = PeftModel.from_pretrained(base_model_fp16, adapter_path)
    
    print("Merging adapter into FP16 model...")
    merged_model = merged_model.merge_and_unload()

    print("\nStarting GGUF conversion process...")
    merged_model_dir = tempfile.mkdtemp()
    
    try:
        merged_model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)
        
        llama_cpp_path = os.environ.get("LLAMA_CPP_PATH")
        if not llama_cpp_path:
            raise ValueError("LLAMA_CPP_PATH environment variable not set. Please set it to the root of your llama.cpp repository.")

        convert_script_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script_path):
            raise FileNotFoundError(f"'convert_hf_to_gguf.py' not found in {llama_cpp_path}")
        
        fp16_gguf_path = args.output_model + ".fp16.gguf"
        cmd_convert = [sys.executable, convert_script_path, merged_model_dir, "--outfile", fp16_gguf_path, "--outtype", "f16"]
        print(f"\nStep 1: Running GGUF conversion (to FP16)...")
        subprocess.run(cmd_convert, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)

        if args.quantization == 16:
            shutil.move(fp16_gguf_path, args.output_model)
        else:
            quantize_executable_name = "llama-quantize"
            # llama.cppのビルドディレクトリを想定
            quantize_executable_path = os.path.join(llama_cpp_path, "build", "bin", quantize_executable_name)
            if not os.path.exists(quantize_executable_path):
                 quantize_executable_path = os.path.join(llama_cpp_path, quantize_executable_name) # ルートにある場合も考慮
            if not os.path.exists(quantize_executable_path):
                raise FileNotFoundError(f"'{quantize_executable_name}' not found. Please build llama.cpp (e.g., by running 'make').")
            
            quant_map = {8: "Q8_0", 4: "Q4_K_M"}
            quant_type = quant_map[args.quantization]
            
            cmd_quantize = [quantize_executable_path, fp16_gguf_path, args.output_model, quant_type]
            print(f"\nStep 2: Running GGUF quantization (to {quant_type})...")
            subprocess.run(cmd_quantize, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
            os.remove(fp16_gguf_path)
        
        print(f"\n✅ Successfully created GGUF model at: {args.output_model}")
        
        create_modelfile(args.base_model, args.output_model)

    except Exception as e:
        print(f"\nAn error occurred during GGUF conversion: {e}")
    finally:
        print("\nCleaning up temporary directories...")
        shutil.rmtree(output_dir)
        shutil.rmtree(merged_model_dir)
        
if __name__ == "__main__":
    main()
