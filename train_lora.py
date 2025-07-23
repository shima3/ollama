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
    """
    指定されたモデルを4-bitでロードし、LoRAの対象となりうるモジュール名を見つけて表示する。
    """
    print(f"Finding targetable modules for {model_name}...")
    try:
        # メモリを節約するため、4-bit量子化でモデルをロード
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )

        # LoRAの対象候補となる線形層のモジュール名を探す
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)):
                module_name = name.split('.')[-1]
                # 一般的なターゲットモジュール名を追加
                if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                    lora_module_names.add(module_name)

        if not lora_module_names:
            print("Could not find any typical targetable modules.")
            print("Please inspect the model architecture manually.")
        else:
            print("\nFound potential target modules for LoRA:")
            print("-----------------------------------------")
            print(", ".join(sorted(list(lora_module_names))))
            print("-----------------------------------------")
            print("Use the --target-modules option to specify which modules to target, e.g.:")
            print(f'--target-modules "{",".join(sorted(list(lora_module_names))[:2])}"')

    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        print("Please ensure the model name is correct and you have access.")

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with LoRA and convert it to GGUF format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "base_model",
        type=str,
        help="The name or path of the base model (e.g., 'meta-llama/Llama-2-7b-hf')."
    )
    parser.add_argument(
        "dataset",
        type=str,
        nargs='?',
        default=None,
        help="Path to the JSON dataset file for fine-tuning. \nIf not provided, the script will only list targetable modules for the base model."
    )
    parser.add_argument(
        "output_model",
        type=str,
        nargs='?',
        default=None,
        help="Path to save the final GGUF model file."
    )
    parser.add_argument(
        "-q", "--quantization",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="Quantization bits for the output GGUF model (4, 8, 16). Default: 8."
    )
    parser.add_argument(
        "-r", "--rank",
        type=int,
        default=8,
        help="LoRA rank. Default: 8."
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor. Default: 16."
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default=None,
        help="Comma-separated list of module names to apply LoRA to (e.g., 'q_proj,v_proj')."
    )

    args = parser.parse_args()

    # モジュール検索モード
    if args.dataset is None:
        find_targetable_modules(args.base_model)
        sys.exit(0)

    # トレーニングモードの引数チェック
    if args.output_model is None:
        parser.error("the following arguments are required: output_model")
    if args.target_modules is None:
        parser.error("the following arguments are required: --target-modules")
        
    # --- 1. モデルとトークナイザーの読み込み ---
    print(f"Loading base model: {args.base_model}")
    
    # 24GB VRAMを考慮し、トレーニングは常に4-bit (QLoRA) で実行
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto", # GPUに自動で配置
        trust_remote_code=True,
    )
    model.config.use_cache = False # トレーニング中はキャッシュを無効化

    # --- 2. データセットの準備 ---
    print(f"Loading and processing dataset: {args.dataset}")
    def format_chat_template(example):
        # JSONが {"messages": [...]} の形式であることを想定
        if isinstance(example.get("messages"), list):
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )}
        return {}

    dataset = load_dataset("json", data_files=args.dataset)["train"]
    processed_dataset = dataset.map(format_chat_template, remove_columns=list(dataset.features))

    # --- 3. LoRAの設 ---
    print("Setting up LoRA configuration...")
    target_modules = [m.strip() for m in args.target_modules.split(',')]
    
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # --- 4. トレーニングの実行 ---
    print("Starting fine-tuning process...")
    # 一時的なチェックポイント保存ディレクトリ
    output_dir = tempfile.mkdtemp()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # 24GB VRAMを考慮
        gradient_accumulation_steps=4, # 実質的なバッチサイズを4に
        learning_rate=2e-4,
        num_train_epochs=1, # エポック数はデータセットサイズに応じて調整
        logging_steps=10,
        fp16=True, # bfloat16が利用可能ならbf16=Trueの方が良い
        save_strategy="no", # チェックポイントは保存しない
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    print("Fine-tuning completed.")

    # --- 5. モデルのマージとGGUFへの変換 ---
    print("Merging LoRA adapter and converting to GGUF...")
    
    # 一時的なマージモデル保存ディレクトリ
    merged_model_dir = tempfile.mkdtemp()

    try:
        # LoRAアダプターをマージ
        merged_model = peft_model.merge_and_unload()
        
        # マージしたモデルとトークナイザーを一時保存
        merged_model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)

        # llama.cppのパスを確認
        llama_cpp_path = os.environ.get("LLAMA_CPP_PATH")
        if not llama_cpp_path or not os.path.exists(os.path.join(llama_cpp_path, "convert.py")):
            raise ValueError(
                "LLAMA_CPP_PATH environment variable not set or convert.py not found in the specified directory.\n"
                "Please set the path to your cloned llama.cpp repository."
            )
        
        convert_script = os.path.join(llama_cpp_path, "convert.py")
        
        # GGUFの量子化タイプを決定
        quant_map = {16: "f16", 8: "q8_0", 4: "q4_K_m"}
        outtype = quant_map[args.quantization]

        # 変換コマンドを実行
        cmd = [
            sys.executable, convert_script, merged_model_dir,
            "--outfile", args.output_model,
            "--outtype", outtype,
        ]
        print(f"\nRunning GGUF conversion command:\n{' '.join(cmd)}\n")
        subprocess.run(cmd, check=True, text=True, capture_output=True)

        print(f"✅ Successfully created GGUF model at: {args.output_model}")

    except Exception as e:
        print(f"An error occurred during GGUF conversion: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print("Stderr:", e.stderr)
            print("Stdout:", e.stdout)
    finally:
        # 一時ディレクトリをクリーンアップ
        shutil.rmtree(output_dir)
        shutil.rmtree(merged_model_dir)
        
if __name__ == "__main__":
    main()
