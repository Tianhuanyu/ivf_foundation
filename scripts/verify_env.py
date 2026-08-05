import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda_matmul_ok:", bool(y.sum().item() != 0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram_free_gb: {free/1e9:.2f} / {total/1e9:.2f}")
