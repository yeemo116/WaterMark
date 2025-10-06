import torch
import time

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(128 * 32 * 32, 10)
).cuda()

dummy = torch.randn(16, 3, 32, 32).cuda()

start = time.time()
for _ in range(100):
    out = model(dummy)
torch.cuda.synchronize()
print("耗時：", time.time() - start)
