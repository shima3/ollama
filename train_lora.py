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

def find_targetable_modules(model_name: str):
    print(f"Finding targetable modules for {model_name}...")
    try:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto', trust_remote_code=True)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)):
                module_name = name.split('.')[-1]
                if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                    lora_module_names.add(module_name)
        if not lora_module_names: print("Could not find any typical targetable modules.")
        else:
            print("\nFound potential target modules for LoRA:")
            print("-----------------------------------------")
            print(", ".join(sorted(list(lora_module_names))))
            print("-----------------------------------------")
            print(f'Use --target-modules, e.g., --target-modules "{",".join(sorted(list(lora_module_names))[:2])}"')
    except Exception as e: print(f"An error occurred: {e}")
    finally:
        if 'model' in locals(): del model
        torch.cuda.empty_cache()

def create_modelfile(base_model_name: str, gguf_path: str):
    """
    指定されたGGUFファイル用のModelfileを自動生成する。
    """
    gguf_filename = os.path.basename(gguf_path)
    modelfile_path = os.path.join(os.path.dirname(gguf_path) or ".", "Modelfile")

    template = ""
    parameters = []

    model_name_lower = base_model_name.lower()
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
        
        # suggested_model_name = os.path.splitext(gguf_filename)[0].replace("-", "_").lower()
        suggested_model_name = os.path.splitext(gguf_filename)[0]
        print("="*70)
        print(f"ollama create {suggested_model_name} -f {modelfile_path}")
        print("="*70)

    except Exception as e:
        print(f"\nCould not create Modelfile: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA and convert to GGUF.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("base_model", type=str, help="Base model name or path.")
    parser.add_argument("dataset", type=str, nargs='?', default=None, help="Dataset JSON file path.")
    parser.add_argument("output_model", type=str, nargs='?', default=None, help="Final GGUF model file path.")
    parser.add_argument("-q", "--quantization", type=int, choices=[4, 8, 16], default=8, help="Quantization bits.")
    parser.add_argument("-r", "--rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--target-modules", type=str, default=None, help="Target modules for LoRA.")
    
    args = parser.parse_args()
    if args.dataset is None:
        find_targetable_modules(args.base_model)
        sys.exit(0)
    if args.output_model is None: parser.error("output_model is required.")
    if args.target_modules is None: parser.error("--target-modules is required.")
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    LLAMA2_CHAT_TEMPLATE = ("{% for message in messages %}{% if message['role'] == 'system' %}{{ '[INST] <<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'user' %}{{ message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}")
    if tokenizer.chat_template is None:
        print("Applying a Llama-2-chat-style template.")
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    
    def preprocess_function(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return tokenizer(text, truncation=True, max_length=2048, padding=False)
    
    dataset = load_dataset("json", data_files=args.dataset)["train"]
    tokenized_dataset = dataset.map(preprocess_function, remove_columns=list(dataset.features))
    
    lora_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=[m.strip() for m in args.target_modules.split(',')], lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    output_dir = tempfile.mkdtemp()
    training_args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4, num_train_epochs=1, logging_steps=10, fp16=True, save_strategy="epoch", report_to="none")
    trainer = Trainer(model=peft_model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    
    print("\nStarting fine-tuning process...")
    trainer.train()
    print("Fine-tuning completed.")
    
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
        if not llama_cpp_path: raise ValueError("LLAMA_CPP_PATH environment variable not set.")

        convert_script_path = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script_path): raise FileNotFoundError(f"'convert_hf_to_gguf.py' not found in {llama_cpp_path}")
        
        fp16_gguf_path = args.output_model + ".fp16.gguf"
        cmd_convert = [sys.executable, convert_script_path, merged_model_dir, "--outfile", fp16_gguf_path, "--outtype", "f16"]
        print(f"\nStep 1: Running GGUF conversion (to FP16)...")
        subprocess.run(cmd_convert, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)

        if args.quantization == 16:
            shutil.move(fp16_gguf_path, args.output_model)
        else:
            quantize_executable_name = "llama-quantize"
            quantize_executable_path = os.path.join(llama_cpp_path, "build", "bin", quantize_executable_name)
            if not os.path.exists(quantize_executable_path):
                raise FileNotFoundError(f"'{quantize_executable_name}' not found. Please build llama.cpp using CMake.")
            
            quant_map = {8: "Q8_0", 4: "Q4_K_M"}
            quant_type = quant_map[args.quantization]
            
            cmd_quantize = [quantize_executable_path, fp16_gguf_path, args.output_model, quant_type]
            print(f"\nStep 2: Running GGUF quantization...")
            subprocess.run(cmd_quantize, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
            os.remove(fp16_gguf_path)
        
        print(f"\n✅ Successfully created GGUF model at: {args.output_model}")
        
        # Modelfileの自動生成
        create_modelfile(args.base_model, args.output_model)

    except Exception as e:
        print(f"\nAn error occurred during GGUF conversion: {e}")
    finally:
        print("\nCleaning up temporary directories...")
        shutil.rmtree(output_dir)
        shutil.rmtree(merged_model_dir)
        
if __name__ == "__main__":
    main()
