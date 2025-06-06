---
hide:
    - toc
---

# Chapter 7: Math Operations

> “Give me a tensor and a math op, and I will move the machine learning world.”

---

## 7.1 Categories of Math Operations in PyTorch

Math in PyTorch is modular and fast. Most tensor operations fall into one of these:

| Category         | Examples                         | Description                                   |
|------------------|----------------------------------|-----------------------------------------------|
| Elementwise Ops  | `+`, `-`, `*`, `/`, `exp`, `log` | Operate on each element independently         |
| Reduction Ops    | `sum`, `mean`, `max`, `prod`     | Reduce one or more dimensions                 |
| Matrix Ops       | `matmul`, `mm`, `bmm`, `einsum`  | Tensor/matrix multiplication and contractions |
| Special Ops      | `clamp`, `abs`, `floor`, `round` | Nonlinear math tricks                         |

Let’s break these down with practical examples.

---

## 7.2 Elementwise Operations

### Basic arithmetic:
```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([3.0, 2.0, 1.0])
a + b
a - b
a * b
a / b
a ** 2
```

### Elementwise functions:
```python
torch.exp(a)
torch.log(a)
torch.sqrt(a)
torch.sin(a)
torch.abs(torch.tensor([-3.0, 2.0]))
```

### In-place versions:
```python
a.add_(1)  # modifies a directly
```
> ⚠️ Use in-place ops (_) with caution in training — they can interfere with autograd.

## 7.3 Reduction Operations
Reductions collapse tensor dimensions into a summary value.
```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
torch.sum(x)            # Sum of all elements
torch.sum(x, dim=0)     # Column-wise sum: tensor([4., 6.])
torch.mean(x)           # Mean
torch.prod(x)           # Product
torch.max(x), torch.min(x)
torch.argmax(x), torch.argmin(x)
```
> You can reduce across specific dimensions with the dim= keyword. This is critical for understanding batch-wise behavior in neural networks.

## 7.4 Logical and Comparison Operations
```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([2.0, 2.0, 2.0])
a == b
a > b
a <= b
a != b
# Use result as mask:
a[a > 2]      # tensor([3.0])
```

### PyTorch also supports:
```python
torch.any(condition)
torch.all(condition)
```

## 7.5 Matrix Operations (Linear Algebra 101)
### Dot Product (1D):
```python
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
torch.dot(a, b)      # 1*3 + 2*4 = 11
```

### Matrix Multiplication:
```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
A @ B                # or torch.matmul(A, B)
```

### Batched Multiplication:
```python
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
torch.bmm(A, B)      # Batch matrix multiply
```

## 7.6 einsum: Einstein Notation
A flexible and expressive way to do tensor contractions, transpositions, reductions.
```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)
torch.einsum('ik,kj->ij', A, B)  # Equivalent to matmul
```
> More readable, chainable, and often better for performance in attention mechanisms and complex models.

##  7.7 Special Math Ops
### Clamping (limit min and max):
```python
x = torch.tensor([0.1, 0.5, 1.5, 3.0])
torch.clamp(x, min=0.2, max=1.0)
```

### Rounding:
```python
torch.floor(x)
torch.ceil(x)
torch.round(x)
```

### Normalization (common in ML):
```python
x = torch.tensor([1.0, 2.0, 3.0])
x_norm = (x - x.mean()) / x.std()
```

## 7.8 Performance Tips
- ✅ Prefer `@` or `matmul()` for clarity and speed.

- ✅ Avoid Python for loops over tensor elements — use broadcasting.

- ✅ Use `.float()` or `.half()` wisely — lower precision = faster compute.

- ⚠️ Avoid in-place ops if you're unsure about autograd compatibility.

##  7.9 Summary
|Type	               |Examples                                |   
|----------------------|----------------------------------------|
|Elementwise	       |     `+`, `*`, `exp`, `log`             |
|Reduction	           | `sum`, `mean`, `max`, `prod`           |
|Matrix	               | `@`, `matmul`, `bmm`, `einsum`         |
|Special Ops	       |     `clamp`, `abs`, `floor`, `round`   |
|                      |                                        |

- PyTorch supports high-performance math via native tensor ops.

- Knowing when to reduce, broadcast, or `matmul` is key to writing efficient models.

- If you're writing anything involving gradients, check for safe usage with autograd.