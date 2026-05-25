import torch

# CPU check
print("CPU check:")
a = torch.tensor([1, 2, 3], device="cpu")
b = torch.tensor([4, 5, 6], device="cpu")
c = torch.add(a, b)
print(f"  cpu: {c}")

# GPU check
if torch.cuda.is_available():
    print("GPU check:")
    a = torch.tensor([1, 2, 3], device="cuda")
    b = torch.tensor([4, 5, 6], device="cuda")
    c = torch.add(a, b)
    print(f"  gpu: {c}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA not available")
