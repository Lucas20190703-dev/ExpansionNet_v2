import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

print(torch.version.cuda)          # Should show your CUDA version
print(torch.cuda.is_available())   # Should be True
print(torch.cuda.nccl.is_available())  # True if NCCL is available