# Chapter 5: Data Types and Devices

> “Precision and placement — the twin pillars of efficient tensor computation.”

---

## 🧬 5.1 `torch.dtype`: The Soul of a Tensor

Every tensor in PyTorch carries a data type that defines the **precision** and **nature** of its values.

### Common `torch.dtype` values:

| `dtype`           | Description                      | Bits |
|-------------------|----------------------------------|------|
| `torch.float32`   | 32-bit floating point (default)  | 32   |
| `torch.float64`   | 64-bit float (aka double)        | 64   |
| `torch.float16`   | 16-bit float (half precision)    | 16   |
| `torch.bfloat16`  | Brain float (used in TPUs)       | 16   |
| `torch.int32`     | 32-bit integer                   | 32   |
| `torch.int64`     | 64-bit integer (aka long)        | 64   |
| `torch.bool`      | Boolean                          | 1    |

### How to specify `dtype`:

```python
x = torch.tensor([1, 2, 3], dtype=torch.float64)
print(x.dtype)  # torch.float64
```

### Casting between types:

```python
x = x.to(torch.float16)
x = x.int()              # Shortcut for int32
x = x.type(torch.float32)
```
> 🔬 float32 is the sweet spot for training: fast and accurate.
But for inference? float16 (or bfloat16) is often enough.

##  5.2 `torch.device`: **The Tensor's Location**

A tensor doesn’t just live in memory — it lives on a device.
```python
cpu_tensor = torch.tensor([1.0])           # On CPU
gpu_tensor = cpu_tensor.to('cuda')         # Moved to GPU
```

You can also create tensors directly on a device:
```python
device = torch.device('cuda')
x = torch.zeros(3, 3, device=device)
```

Detecting and using available GPUs:
```python
if torch.cuda.is_available():
    print("CUDA ready! Let's party.")
else:
    print("Stuck on CPU. Meh.")
```
> 💡 For multi-GPU systems: use 'cuda:0', 'cuda:1', etc.

## 5.3 Default Data Type Settings

Sometimes you want to globally change the default dtype. PyTorch lets you do this:
```python
torch.set_default_dtype(torch.float64)
```
This affects all future float tensors created via:
```python
torch.zeros(3)      # Now float64
torch.tensor([1.0]) # Now float64
```
Check the current default:
```python
torch.get_default_dtype()
```
> Useful when training scientific models (need precision)
or optimizing inference (want float16).

## 5.4 Intro to Mixed Precision
In modern deep learning (especially with GPUs like RTX, A100, etc.), mixed precision is the name of the game.

### What’s Mixed Precision?
Training with both:

- `float32` (for critical values like loss gradients)

- `float16` or `bfloat16` (for speed and memory savings)

### Why use it?
- 🚄 Faster training with Tensor Cores

- 📉 Less memory usage = bigger models

### How to use it?
Start with PyTorch’s AMP (Automatic Mixed Precision):

```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
    loss = criterion(output, target)
```
> ⚠️ We’ll cover this fully in Chapter 17: Using Torch with CUDA.
For now, just know it’s a killer optimization strategy.

## 5.5 Summary
- `torch.dtype` controls a tensor’s precision and data interpretation.

- `torch.device` decides where the tensor lives — CPU or GPU.

- You can set default dtypes, move tensors across devices, and leverage mixed precision for high-performance computing.

- These two concepts are subtle but powerful when optimizing both training and inference.

