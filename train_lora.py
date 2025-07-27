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

# create_modelfile関数 (戻り値を削除)
def create_modelfile(base_model_name: str, tokenizer: AutoTokenizer):
    gguf_filename = "model.gguf"
    modelfile_path = "Modelfile"
    template, parameters = "", []
    model_name_lower = base_model_name.lower()
    # (テンプレート判定ロジックは変更なし)
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        if "[INST] <<SYS>>" in tokenizer.chat_template: template, parameters = 'TEMPLATE """[INST] <<SYS>>\n{{ .System }}\n<</SYS>>\n\n{{ .Prompt }} [/INST]"""', ['PARAMETER stop "[INST]"', 'PARAMETER stop "[/INST]"']
        elif "user" in tokenizer.chat_template and "assistant" in tokenizer.chat_template and "<|start_header_id|>" in tokenizer.chat_template: template, parameters = 'TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""', ['PARAMETER stop "<|eot_id|>"', 'PARAMETER stop "<|end_header_id|>"']
        elif "[INST]" in tokenizer.chat_template and "[/INST]" in tokenizer.chat_template: template, parameters = 'TEMPLATE """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]"""', ['PARAMETER stop "[INST]"', 'PARAMETER stop "[/INST]"']
    if not template:
        if "llama-3" in model_name_lower: template, parameters = 'TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""', ['PARAMETER stop "<|eot_id|>"', 'PARAMETER stop "<|end_header_id|>"']
        elif "llama-2" in model_name_lower: template, parameters = 'TEMPLATE """[INST] <<SYS>>\n{{ .System }}\n<</SYS>>\n\n{{ .Prompt }} [/INST]"""', ['PARAMETER stop "[INST]"', 'PARAMETER stop "[/INST]"']
        elif "mistral" in model_name_lower or "codellama" in model_name_lower: template, parameters = 'TEMPLATE """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]"""', ['PARAMETER stop "[INST]"', 'PARAMETER stop "[/INST]"']
    
    content = f"FROM ./{gguf_filename}\n"
    if template: content += f"\n{template.strip()}\n"
    if parameters: content += "\n" + "\n".join(parameters) + "\n"
    content += "\nPARAMETER temperature 0.7\n"
    with open(modelfile_path, "w", encoding="utf-8") as f: f.write(content)
    print(f"✅ Successfully created Modelfile at: ./{modelfile_path}")

# generate_training_info関数 (役割を明確化)
def generate_training_info(args: argparse.Namespace):
    info_path = "training_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Training Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*40 + "\n")
        f.write("Parameters (from Python script):\n")
        f.write(f"  base_model: {args.base_model}\n")
        f.write(f"  dataset_used: dataset.json\n")
        f.write(f"  target_modules: {args.target_modules}\n")
        f.write(f"  quantization: {args.quantization}\n")
        f.write(f"  rank: {args.rank}\n")
        f.write(f"  alpha: {args.alpha}\n")
        f.write(f"  epoch: {args.epoch}\n")
        f.write("="*40 + "\n")
    print(f"✅ Successfully created training info file at: ./{info_path}")

def main():
    parser = argparse.ArgumentParser(description="Model building engine for LoRA fine-tuning.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("base_model", type=str, help="Base model name or path.")
    parser.add_argument("--target-modules", type=str, required=True, help="Comma-separated list of target modules for LoRA.")
    parser.add_argument("-q", "--quantization", type=int, choices=[4, 8, 16], default=4, help="Quantization bits. Default: 4.")
    parser.add_argument("-r", "--rank", type=int, default=16, help="LoRA rank. Default: 16.")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha. Default: 32.")
    parser.add_argument("-e", "--epoch", type=int, default=5, help="Number of training epochs. Default: 5.")
    args = parser.parse_args()

    final_gguf_path = "model.gguf"
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # チャットテンプレートが設定されていなければ、強制的に設定する。
    if tokenizer.chat_template is None:
        print("Tokenizer `chat_template` is not set. Applying ELYZA/Codellama template manually.")
        chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + eos_token }}"
                "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.chat_template = chat_template

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    
    dataset = load_dataset("json", data_files="dataset.json")["train"]
    def preprocess_function(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=2048)
    tokenized_dataset = dataset.map(preprocess_function, remove_columns=list(dataset.features))
    lora_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=[m.strip() for m in args.target_modules.split(',')], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    training_output_dir = tempfile.mkdtemp()
    training_args = TrainingArguments(output_dir=training_output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4, num_train_epochs=args.epoch, logging_steps=10, fp16=True, save_strategy="epoch", report_to="none")
    trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized_dataset, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
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
    try:
        merged_model.save_pretrained(merged_model_dir); tokenizer.save_pretrained(merged_model_dir)
        llama_cpp_path = os.environ.get("LLAMA_CPP_PATH")
        if not llama_cpp_path: raise ValueError("LLAMA_CPP_PATH environment variable not set.")
        convert_script_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script_path): raise FileNotFoundError(f"'convert_hf_to_gguf.py' not found in {llama_cpp_path}")
        fp16_gguf_path = "model-fp16-temp.gguf"
        subprocess.run([sys.executable, convert_script_path, merged_model_dir, "--outfile", fp16_gguf_path, "--outtype", "f16"], check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
        if args.quantization == 16:
            shutil.move(fp16_gguf_path, final_gguf_path)
        else:
            quantize_exec_name = "llama-quantize"
            quantize_exec_path = os.path.join(llama_cpp_path, "build", "bin", quantize_exec_name)
            if not os.path.exists(quantize_exec_path): quantize_exec_path = os.path.join(llama_cpp_path, quantize_exec_name)
            if not os.path.exists(quantize_exec_path): raise FileNotFoundError(f"'{quantize_exec_name}' not found. Please build llama.cpp.")
            quant_map_cli = {8: "Q8_0", 4: "Q4_K_M"}
            quant_type_cli = quant_map_cli[args.quantization]
            print(f"\nStep 2: Running GGUF quantization (to {quant_type_cli})...")
            subprocess.run([quantize_exec_path, fp16_gguf_path, final_gguf_path, quant_type_cli], check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
            os.remove(fp16_gguf_path)
        
        print(f"\n✅ Successfully created GGUF model at: ./{final_gguf_path}")
        
        # --- 変更点: 呼び出す関数をシンプル化 ---
        create_modelfile(args.base_model, tokenizer)
        generate_training_info(args)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
    finally:
        shutil.rmtree(training_output_dir)
        shutil.rmtree(merged_model_dir)

if __name__ == "__main__":
    main()
