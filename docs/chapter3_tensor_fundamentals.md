---
hide:
    - toc
---

# Chapter 3: Tensor Fundamentals

> “Before we reshape the world with tensors, let’s understand what they truly are.”

---

## 🔍 3.1 What Is a Tensor?

A tensor is the core data structure in PyTorch. It’s a multi-dimensional array that represents data — and supports a wide range of mathematical operations on CPUs or GPUs.

But that definition alone is too dry. Let’s break it down by dimensions:

| Tensor Dim | Common Name       | Example                  | Shape    |
|------------|-------------------|--------------------------|----------|
| 0D         | Scalar            | `42`                     | `()`     |
| 1D         | Vector            | `[1, 2, 3]`              | `(3,)`   |
| 2D         | Matrix            | `[[1, 2], [3, 4]]`       | `(2, 2)` |
| 3D+        | Tensor (generic)  | Batch of images, video   | `(B, C, H, W)` |

In PyTorch:

```python
import torch
x = torch.tensor([[1, 2], [3, 4]])
```
This creates a 2×2 matrix — a rank-2 tensor.
All machine learning models, no matter how complex, are just functions that operate on tensors.

## 3.2 Tensors vs NumPy Arrays

|Feature	                |NumPy`ndarray`	                  |PyTorch`Tensor`      |
|---------------------------|---------------------------------|---------------------|
|GPU support	            |❌ CPU only	                    |✅ CUDA, MPS, etc.   |
|Autograd	                |❌ No automatic gradients	    |✅ Built-in autograd |
|Deep learning ready	    |❌ Needs integration	        |✅ Native support    |
|API similarity	            |✅ High	                        |✅ Nearly identical  |
|Interoperability	        |✅ torch.from_numpy() / .numpy()|	                   |
|                           |                                 |                     | 

PyTorch was designed to mimic NumPy’s API — so if you’ve written code in NumPy before, tensors will feel familiar. But they also bring GPU acceleration and automatic differentiation into the mix.
```python
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])
torch_tensor = torch.from_numpy(np_arr)
# Back to NumPy
np_back = torch_tensor.numpy()
```
> ⚠️ **Note:** Both share the same memory buffer unless you .clone() the tensor.

## 3.3 Device Management: CPU and CUDA

By default, PyTorch tensors live on the CPU. But the magic happens when you move them to the GPU (via CUDA).
**Check if CUDA is available:**
```python
torch.cuda.is_available()  # Returns True if you have a compatible GPU
```
**Move tensor to CUDA:**
```python
x = torch.tensor([1.0, 2.0])
x = x.to('cuda')        # Send to GPU
x = x.to('cpu')         # Bring back to CPU
```
You can also be explicit:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1.0, 2.0], device=device)
```

## 3.4 CPU vs GPU: When It Matters

|Use Case	                |CPU	                      |GPU                  |
|---------------------------|-----------------------------|---------------------|
|Small models or inference	|✅ Fine	                    |✅ Maybe overkill     |
|Deep neural nets	        |❌ Too slow	                |✅ Ideal               |
|Training loops w/ backprop	|❌ Bottleneck	            |✅ Accelerated         |
|Parallel data ops	        |✅ w/ multiprocessing	    |✅ Massively parallel  |
|                           |                            |                         |

> 💡 Even simple tensor math is often 10–100× faster on the GPU.

## 3.5 Summary

- A tensor is a generalization of scalars, vectors, and matrices.

- PyTorch’s Tensor is like NumPy’s ndarray, but with GPU and autograd support.

- Devices (cpu or cuda) matter — especially for training speed.

- Tensor creation is simple — but choosing the right device is crucial for performance.