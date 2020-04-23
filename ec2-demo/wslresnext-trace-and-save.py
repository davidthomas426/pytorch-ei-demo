import torch

image = torch.rand(1, 3, 224, 224)

# eval() toggles inference mode
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').eval()

# Compile model to TorchScript via tracing
with torch.no_grad():
  # You can trace with any input
  traced_model = torch.jit.trace(model, image)

# Serialize traced model
torch.jit.save(traced_model, 'wslresnext101_32x8d_traced.pt')
