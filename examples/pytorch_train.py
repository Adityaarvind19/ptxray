"""A small PyTorch training loop with two classic performance mistakes, to show
what cudatrace sees inside a real framework:

  * the input batch is copied from pageable host memory every step, and
  * loss.item() forces a GPU->host sync every step.

Run:  ./bin/cudatrace -o pytorch.json -- python examples/pytorch_train.py
Then: --fixed  removes both mistakes (pinned memory + logging every 50 steps).

Note: cuBLAS/cuDNN launch their kernels through the CUDA driver API, which
runtime-API interposition does not see; PyTorch's own kernels (elementwise,
reductions, optimizer) and all its memory traffic are captured.
"""
import sys
import time

import torch

fixed = "--fixed" in sys.argv
torch.manual_seed(0)
dev = "cuda"
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048), torch.nn.ReLU(),
    torch.nn.Linear(2048, 2048), torch.nn.ReLU(),
    torch.nn.Linear(2048, 10),
).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
x_host = torch.randn(256, 1024)
y_host = torch.randint(0, 10, (256,))
if fixed:
    x_host, y_host = x_host.pin_memory(), y_host.pin_memory()

t0 = time.perf_counter()
for step in range(300):
    x = x_host.to(dev, non_blocking=fixed)
    y = y_host.to(dev, non_blocking=fixed)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if not fixed or step % 50 == 0:
        last = loss.item()  # GPU -> host sync
torch.cuda.synchronize()
print(f"{'fixed' if fixed else 'naive'}: 300 steps in {time.perf_counter() - t0:.3f} s, loss {last:.4f}")
