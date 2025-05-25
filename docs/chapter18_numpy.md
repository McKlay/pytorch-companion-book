# Chapter 18: Integration with NumPy

> “Two libraries. One memory space. No nonsense.”

---

## 18.1 Why Integrate with NumPy?

**NumPy** is the OG of numerical computing in Python. Even if you're deep into PyTorch, you’ll often need to:

- Preprocess data with NumPy  
- Use NumPy functions not in PyTorch  
- Interface with external libs (e.g., OpenCV, SciPy, Pandas)  
- Visualize tensors using matplotlib or seaborn  

> PyTorch makes this easy by letting you **share memory** between `torch.Tensor` and `np.ndarray`.

---

## 18.2 `torch.from_numpy()` — NumPy ➜ Tensor

```python
import numpy as np
import torch

arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)

arr[0] = 99
print(t)  # tensor([99., 2., 3.])
```
>  This is fast and efficient — no memory copy.

---

##  18.3 .numpy() — Tensor ➜ NumPy
```python
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()
```
Again, same memory — not a copy.

### 🛑 But you must be on the CPU:
```python
t = torch.tensor([1.0, 2.0]).to('cuda')
arr = t.cpu().numpy()  # Must move to CPU first!
```

---

##  18.4 Use Case Examples

### ➤ Matplotlib visualization:
```python
import matplotlib.pyplot as plt
image = torch.randn(28, 28)
plt.imshow(image.numpy(), cmap='gray')
```
### ➤ Pandas & CSV I/O:
```python
import pandas as pd
df = pd.read_csv('data.csv')
tensor = torch.from_numpy(df.values).float()
```
### ➤ Feature extraction / math ops:
```python
np.mean(tensor.numpy(), axis=0)
```
Use NumPy when you need broadcasting or functions that PyTorch lacks (e.g., np.percentile()).


---

## 18.5 Zero-Copy Interoperability

The NumPy↔Torch conversion is zero-copy. That’s great, but watch for:

|Gotcha	                                |Solution                               |
|---------------------------------------|---------------------------------------|
|CUDA tensors can't `.numpy()`	        |Move to CPU first: t.cpu()`.numpy() `    |
|Detached views may be unsafe	        |Use `.clone()` if unsure                 |
|Mixed float types (e.g. float64)	    |Use `.float()` before model usage        |
|In-place ops affect both	            |Clone before modifying either          |


---

## 🚫 18.6 When Not to Use `.numpy()`

Avoid inside:

- Training loops — breaks CUDA pipelines and slows performance

- Tensors requiring gradients — `.numpy()` drops the computation graph

- GPU batches — unnecessary overhead

Use `.numpy()` only for:

- Visualization

- Evaluation

- Exporting data

- Debugging

---

## 18.7 Tips for Interfacing with NumPy

- Use `.detach().cpu().numpy()` to safely extract predictions:

```python
preds = model(x)
np_preds = preds.detach().cpu().numpy()
```

- Convert back to `float32` if NumPy defaults to `float64`:
```python
torch.from_numpy(arr.astype(np.float32))
```

- Use `.contiguous()` if you hit **weird shape bugs**:
```python
tensor = tensor.permute(1, 0).contiguous()
```

---

## 18.8 Summary

|Direction	                |Method                     |
|---------------------------|---------------------------|
|NumPy → Tensor	            |`torch.from_numpy()`         |
|Tensor → NumPy	            |`.numpy() `                  |
|Ensure CPU-only	        |`.cpu().numpy()`             |
|Keep gradients safe	    |`.detach().cpu().numpy()`    |

- PyTorch and NumPy **play together beautifully** — just watch out for **GPU vs CPU** boundaries

- These conversions are **efficient**, but must be used with care during **training**