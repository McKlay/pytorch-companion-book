# Chapter 4: `torch.Tensor`

> “The universe of PyTorch begins with a single tensor—and everything else builds on top of it.”

---

## 4.1 Tensor Creation Methods

Let’s explore how to instantiate tensors like a boss. PyTorch provides multiple ways depending on your use case:

### ▶ Basic Constructor
```python
torch.tensor([1, 2, 3])                 # From Python list
torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2D Float Tensor
```
### ▶ Pre-filled Factories
```python
torch.zeros(2, 3)           # All zeros
torch.ones(2, 3)            # All ones
torch.full((2, 2), 42)      # All elements are 42
torch.eye(3)                # Identity matrix
torch.arange(0, 10, 2)      # Like Python’s range()
torch.linspace(0, 1, 5)     # 5 values between 0 and 1
```
### ▶ Like Another Tensor
```python
x = torch.ones(2, 2)
torch.zeros_like(x)
torch.rand_like(x)
```
## 📐 4.2 Tensor Properties
Every tensor has a few critical attributes:  
**🔸 shape and size()**
```python
x.shape            # torch.Size([2, 3])
x.size()           # Same as above
```
**🔸 dtype – data type**
```python
x.dtype            # e.g., torch.float32
x = x.to(torch.int64)  # change type
```
**🔸 device – CPU or CUDA**
```python
x.device           # Shows current device
x = x.to('cuda')   # Move to GPU
```
**🔸 requires_grad**
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
x.requires_grad    # True
```
---

## 4.3 Tensor Operations
PyTorch supports extensive elementwise operations directly on tensors:

**Arithmetic**
```python
x + y
x - y
x * y
x / y
x ** 2
```
**Comparison**
```python
x == y
x > y
x != y
```
**Logical**
```python
torch.logical_and(x > 0, x < 1)
```
> ✅ These are vectorized — no need for loops!

---

## 4.4 Reshaping & Reorganizing Tensors
These tools let you morph tensor shapes without changing their content.

**Reshape & View**
```python
x.view(-1)           # Flatten (requires contiguous memory)
x.reshape(2, 3)      # Flexible reshape
```
**Squeeze & Unsqueeze**
```python
x = torch.zeros(1, 3, 1)
x.squeeze()          # Remove dim=1 → (3,)
x.unsqueeze(0)       # Add dim at index 0 → (1, 3, 1)
```
**Permute & Transpose**
```python
x = torch.randn(2, 3, 4)
x.permute(2, 0, 1)   # Reorders dimensions
x.transpose(0, 1)    # Swaps two dims only
```
> 🔁 Use permute() for high-dimensional tensors (images, etc.)

## 4.5 Indexing & Slicing
Basic and advanced ways to access tensor values:
```python
x[0]             # First row
x[:, 1]          # Second column
x[1:, :]         # All rows except the first
```
Also supports:
```python
x[x > 0]                         # Boolean mask
x[torch.tensor([0, 2])]          # Indexing with tensor
```
> These are the same ideas as NumPy — but with GPU support.

## 4.6 Autograd Compatibility

One of PyTorch’s killer features is automatic differentiation — made possible because every Tensor can carry its computation history.
```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3
z = y.sum()
z.backward()
x.grad  # ∂z/∂x
```
Don’t track:
```python
with torch.no_grad():
    result = model(x)
```
> 🔥 If requires_grad=True, the tensor is part of the computation graph. Perfect for training neural nets.

## 4.7 Miscellaneous API Tricks

**Clone vs Detach:**
```python
x.clone()        # Returns a copy
x.detach()       # Returns a tensor with no grad-tracking
```
**Checking if a tensor is contiguous:**
```python
x.is_contiguous()
```
> Useful when using .view() which demands contiguous memory layout.

## 4.8 Summary

- `torch.Tensor` is more than just an array — it has memory, gradient, and device awareness.

- You can create tensors using many constructors (`zeros`, `arange`, `full`, etc.)

- Tensors support rich operations: math, reshape, slice, compare, move to CUDA.

- They're also autograd-aware — making them perfect for deep learning.


