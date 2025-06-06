---
hide:
    - toc
---

# Chapter 17: Using `torch` with CUDA

> “If your tensors aren’t on the GPU, are they even lifting?”

---

## 17.1 What is CUDA?

CUDA stands for **Compute Unified Device Architecture** — NVIDIA’s parallel computing platform.

In PyTorch, it means:

- Massive speedups via **GPU acceleration**  
- Easy-to-use APIs to move computation to CUDA  
- Seamless switching between CPU and GPU  

> PyTorch abstracts CUDA beautifully. If you can use `.to('cuda')`, **you can GPU**.

---

## 17.2 Check CUDA Availability

Before using CUDA, always check:

```python
import torch
torch.cuda.is_available()  # Returns True if CUDA is ready
torch.cuda.device_count()  # Number of available GPUs
```

---

### 17.3 Setting Your Device
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(5, 5).to(device)
model = model.to(device)
```
> You can also specify GPU index: `'cuda:0'`, `'cuda:1'`, etc.

---

## 17.4 Moving Data to and from GPU
```python
x = torch.tensor([1.0, 2.0])
x_cuda = x.to('cuda')

# Back to CPU
x_cpu = x_cuda.to('cpu')
```
> ⚠️ Tensors must be on the same device for math to work. <br>
❌ CPU-GPU ops will crash with RuntimeError.

---

##  17.5 Multi-GPU Usage

### ➤ List all GPUs:
```python
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
```

### ➤ Move model to a specific GPU:
```python
model = model.to('cuda:1')
```

### ➤ Use DataParallel (basic multi-GPU training):
```python
from torch.nn import DataParallel

model = DataParallel(model)
model = model.to('cuda')
```
> ✅ Automatically splits input batches <br>
For large-scale training, DistributedDataParallel is preferred (coming in advanced chapters)

---

## 17.6 Memory Management and Stats

### ➤ Track VRAM usage:
```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```

### ➤ Free unused memory:
```python
torch.cuda.empty_cache()
```
> This won’t free memory from PyTorch internally, but it makes it available to other applications.

---

##  17.7 Common CUDA Pitfalls

|Pitfall	                                |Fix                                                |
|-------------------------------------------|---------------------------------------------------|
|Mixing CPU and GPU tensors	                |`.to(device)` both inputs before operations          |
|Forgetting `.to(device)` on model	        |Model stays on CPU → loss never goes down          |
|Out of memory (OOM)	                    |Reduce batch size or use with `torch.no_grad()`      |
|CUDA slower than CPU (tiny model)	        |CUDA overhead may outweigh benefits                |
|GPU idle, CPU overloaded	                |Use `num_workers` in DataLoader + pin_memory         |


---

## 17.8 AMP (Automatic Mixed Precision)

For faster training with less memory usage:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for input, target in dataloader:
    optimizer.zero_grad()
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
> AMP = ⚡ Speed + 💾 Efficiency without major code rewrites.


---

## 17.9 Benchmark Settings (CuDNN)
```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```
- Set `benchmark=True` to let PyTorch auto-optimize conv performance

- Set `deterministic=True` if you need exact reproducibility

---

## 17.10 Summary

|Action	                    |Code Example                          |
|---------------------------|--------------------------------------|
|Set device	                |device = torch.device("cuda")         |
|Move tensor/model	        |.to(device)                           |
|Multi-GPU (basic)	        |torch.nn.DataParallel(model)          |
|Monitor memory usage	    |memory_allocated(), empty_cache()     |
|Mixed precision training	|torch.cuda.amp.autocast()             |

-  GPUs = speed — use them wisely

- .to(device) is your best friend — for tensors, models, inputs, and labels

- Track your memory, use AMP for large models, and never mix CPU + CUDA in a single operation