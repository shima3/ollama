#!/bin/bash

# スクリプトが失敗した場合に実行を停止
set -e

# 使用方法を表示する関数
usage() {
  echo "Usage: $0 <output_dir> <base_model> <dataset> --target-modules \"<modules>\" [OPTIONS]"
  echo
  echo "This script automates the entire process: training, GGUF conversion,"
  echo "and creating the final Ollama model."
  echo
  echo "Arguments:"
  echo "  <output_dir>          Directory for artifacts. Renames old one if exists."
  echo "  <base_model>          Base model name or path."
  echo "  <dataset>             Path to the training dataset JSON file."
  echo
  echo "Required Options:"
  echo "  --target-modules \"...\"  Comma-separated list of target modules for LoRA."
  exit 1
}

# 引数の数をチェック
if [ "$#" -lt 4 ]; then
  usage
fi

OUTPUT_DIR="$1"
BASE_MODEL="$2"
ORIGINAL_DATASET_PATH="$3"

# 既存ディレクトリのアーカイブ処理
if [ -d "$OUTPUT_DIR" ]; then
    echo "Warning: Directory '$OUTPUT_DIR' already exists. It will be renamed."
    TIMESTAMP_SUFFIX=""
    if [ "$(uname)" == "Darwin" ]; then CREATION_TIMESTAMP=$(stat -f %B "$OUTPUT_DIR"); TIMESTAMP_SUFFIX=$(date -r "$CREATION_TIMESTAMP" +'%Y%m%d-%H%M%S');
    elif [ "$(uname)" == "Linux" ]; then
        if stat -c %W "$OUTPUT_DIR" > /dev/null 2>&1 && [ "$(stat -c %W "$OUTPUT_DIR")" != "0" ]; then CREATION_TIME_STR=$(stat -c %w "$OUTPUT_DIR"); else echo "  (Note: Birth time not available. Using modification time.)"; CREATION_TIME_STR=$(stat -c %y "$OUTPUT_DIR"); fi
        TIMESTAMP_SUFFIX=$(date -d "$CREATION_TIME_STR" +'%Y%m%d-%H%M%S'); fi
    if [ -n "$TIMESTAMP_SUFFIX" ]; then RENAMED_DIR="${OUTPUT_DIR}-${TIMESTAMP_SUFFIX}"; echo "         Renaming existing directory to '$RENAMED_DIR'"; mv "$OUTPUT_DIR" "$RENAMED_DIR"; else echo "Error: Could not determine directory timestamp to rename."; exit 1; fi
fi

if [ ! -f "$ORIGINAL_DATASET_PATH" ]; then echo "Error: Dataset file not found at '$ORIGINAL_DATASET_PATH'"; exit 1; fi

echo -e "\nCreating new output directory '$OUTPUT_DIR'..."
mkdir -p "$OUTPUT_DIR"

# データセットの配置 (ハードリンク or コピー)
TARGET_DATASET_PATH="$OUTPUT_DIR/dataset.json"
echo -e "\nPlacing dataset into output directory as 'dataset.json'..."
if ln "$ORIGINAL_DATASET_PATH" "$TARGET_DATASET_PATH" 2>/dev/null; then echo "  ✅ Successfully created hard link."; else echo "  ⚠️ Hard link failed. Falling back to copy."; cp "$ORIGINAL_DATASET_PATH" "$TARGET_DATASET_PATH"; echo "  ✅ Successfully copied dataset."; fi
ABS_ORIGINAL_DATASET_PATH=$(cd "$(dirname "$ORIGINAL_DATASET_PATH")" && pwd)/$(basename "$ORIGINAL_DATASET_PATH")

shift 3

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_SCRIPT="$SCRIPT_DIR/train_lora.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: train_lora.py not found."; exit 1; fi

echo "=================================================="
echo "Output Directory:   $OUTPUT_DIR"
echo "Base model:         $BASE_MODEL"
echo "Original Dataset:   $ABS_ORIGINAL_DATASET_PATH"
echo "Other arguments:    $@"
echo "=================================================="

cd "$OUTPUT_DIR"

echo -e "\n--- Step 1: Executing Python script for model building ---"
if command -v python3 &> /dev/null; then
    python3 "$PYTHON_SCRIPT" "$BASE_MODEL" "$@"
else
    python "$PYTHON_SCRIPT" "$BASE_MODEL" "$@"
fi
echo "--- Python script finished ---"

echo -e "\n--- Step 2: Finalizing training_info.txt ---"
echo "  dataset_original_path: $ABS_ORIGINAL_DATASET_PATH" >> training_info.txt
echo "✅ Appended original dataset path to training_info.txt"

# --- ▼▼▼ ここからが新しいロジック ▼▼▼ ---
echo -e "\n--- Step 3: Creating and executing Ollama model script ---"
MODEL_TAG="$(basename "$(pwd)"):latest"

# create_ollama_model.sh をこのシェルスクリプトが作成する
cat > create_ollama_model.sh << EOF
#!/bin/sh
# This script was auto-generated by train_lora.sh
cd "\$(dirname "\$0")"
ollama create $MODEL_TAG -f Modelfile
echo "✅ Ollama model '$MODEL_TAG' created successfully."
EOF
chmod +x create_ollama_model.sh

echo "✅ Created executable script: ./create_ollama_model.sh"
echo "Executing it now to create model: $MODEL_TAG"

# 作成したスクリプトを即座に実行
./create_ollama_model.sh
# --- ▲▲▲ 新しいロジックここまで ▲▲▲ ---

echo -e "\n🎉 All processes completed successfully!"
if [ -n "$RENAMED_DIR" ]; then
    echo "The previous run was archived to '${RENAMED_DIR}'."
fi