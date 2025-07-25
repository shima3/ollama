import argparse
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb

def find_targetable_modules(model_name: str):
    """
    指定されたHugging Faceモデルをロードし、LoRAの対象となりうるモジュールを特定して表示する。

    Args:
        model_name (str): Hugging Face Hub上のモデル名またはローカルパス。
    """
    print(f"Finding targetable modules for {model_name}...")
    try:
        # 4-bit量子化でモデルをロードしてメモリ使用量を削減
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True
        )
        
        lora_module_names = set()
        # 一般的なLoRA対象モジュール名のリスト
        typical_modules = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}

        # モデルのすべてのモジュールを探索
        for name, module in model.named_modules():
            # 量子化されたリニアレイヤー、または通常のリニアレイヤーが対象
            if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear)):
                # モジュール名の末尾部分を取得（例: model.layers.0.self_attn.q_proj -> q_proj）
                module_name = name.split('.')[-1]
                if module_name in typical_modules:
                    lora_module_names.add(module_name)
        
        if not lora_module_names:
            print("Could not find any typical targetable modules.")
            print("You may need to inspect the model architecture manually.")
        else:
            sorted_modules = sorted(list(lora_module_names))
            print("\nFound potential target modules for LoRA:")
            print("-----------------------------------------")
            print(", ".join(sorted_modules))
            print("-----------------------------------------")
            print(f"Use the --target-modules argument in the training script, for example:")
            print(f'--target-modules "{",".join(sorted_modules[:2])}"')

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # メモリを解放
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description="Find targetable modules for LoRA in a given Hugging Face model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model_name", type=str, help="The name or path of the model to inspect (e.g., 'meta-llama/Llama-2-7b-chat-hf').")
    args = parser.parse_args()
    
    find_targetable_modules(args.model_name)

if __name__ == "__main__":
    main()
