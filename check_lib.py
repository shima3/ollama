import torch

# 1. PyTorchのバージョンを確認
print(f"PyTorch Version: {torch.__version__}")

# 2. CUDAが利用可能かを確認 (Trueと表示されれば成功)
is_available = torch.cuda.is_available()
print(f"CUDA is available: {is_available}")

# 3. もし利用可能な場合、PyTorchがどのCUDAバージョンでビルドされたかを確認
if is_available:
    print(f"PyTorch was built with CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")