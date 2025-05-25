# Chapter 8: Broadcasting and Shape Ops

> “Shape your tensors, or they will shape your debugging sessions.”

---

## 8.1 What is Broadcasting?

Broadcasting lets PyTorch perform arithmetic operations on tensors of different shapes **without copying or expanding data manually**.

> Imagine it as *virtual expansion* — PyTorch stretches the smaller tensor across the bigger one without allocating new memory.

### Example:

```python
a = torch.tensor([[1], [2], [3]])   # Shape: (3, 1)
b = torch.tensor([10, 20])         # Shape: (2,)
c = a + b                          # Shape: (3, 2)

Here’s what PyTorch imagines behind the scenes:
```lua
a = [[1],    b = [10, 20]   →   [[1+10, 1+20],
     [2],                      [2+10, 2+20],
     [3]]                     [3+10, 3+20]]
```
> No manual tiling. No sweat.

## 8.2 Broadcasting Rules
To broadcast two tensors:

1. Start from the trailing dimensions (i.e., compare right to left).

2. Dimensions must be:

    - Equal, OR

    - One of them is 1, OR

    - One is missing (implied 1)

```lua
Shape A	    Shape B	    Result Shape	    Valid?

(3, 1)	    (1, 4)	    (3, 4)	            ✅

(2, 3)	    (3,)	    (2, 3)	            ✅

(2, 3)	    (3, 2)	    ❌	               ❌

```


## 8.3 Shape Ops You Must Know
These are the reshape tools every PyTorch practitioner must master.

### 🔹 `reshape()` vs `view()`
```python
x = torch.arange(6)        # [0, 1, 2, 3, 4, 5]
x.reshape(2, 3)            # OK anytime
x.view(2, 3)               # Only if x is contiguous
```
> `reshape()` is `safer`, `view()` is faster but stricter.
---

### 🔹 `squeeze()` and `unsqueeze()`
- `squeeze()` removes dimensions of size 1
- `nsqueeze(dim)` adds a 1-sized dimension at position `dim`
```python
x = torch.zeros(1, 3, 1)
x.squeeze()       # shape: (3,)
x.unsqueeze(0)    # shape: (1, 1, 3, 1)
```
> Essential for converting between batch and single-item tensors.
---

### 🔹 `expand()` vs `repeat()`
Both make a tensor appear larger — but in **very different ways**.
- `expand(`): No memory copy. Just a view.
```python
x = torch.tensor([[1], [2]])
x.expand(2, 3)  # OK: repeats the column virtually
```
- `repeat()`: Physically copies data.
```python
x.repeat(1, 3)   # Actually allocates more memory
```
> ✅ Use expand() when possible. It’s faster and leaner.
---

### 🔹 `permute()` and `transpose()`
- permute() — changes *any* dimension order
```python
x = torch.randn(2, 3, 4)
x.permute(2, 0, 1)  # new shape: (4, 2, 3)
```

- `transpose(dim0, dim1)` — swaps two dimensions
```python
x.transpose(0, 1)
```
> Use `permute()` for more complex reordering (e.g., images → channels-first/last).

## 8.4 Real-World Use Cases

| Task                              | Operation Needed               |
|-----------------------------------|--------------------------------|
| Convert grayscale to batch        | `unsqueeze(0)`                 |
| Flatten a CNN layer output        | `.view(batch_size, -1)`        |
| Add channel dim to image          | `unsqueeze(0)` or `permute()`  |
| Match label shapes for loss       | `squeeze()`                    |
| Expand bias term in matmul        | `expand()`                     |



##  8.5 Common Pitfalls
- Incompatible shapes: Use `.shape` to debug before applying ops.

- `view()` on non-contiguous tensors: Use `.contiguous()` or switch to `reshape()`.

- Unintended broadcasting: Always print tensor shapes if math results look suspicious.

## 8.6 Summary
- Broadcasting enables operations on mismatched shapes.

- Reshape tools like `view`, `reshape`, `squeeze`, and `unsqueeze` give full control over dimensions.

- `expand()` is fast and memory-efficient — use it over `repeat()` when possible.

- Shape ops are essential for building models, writing clean data pipelines, and debugging runtime errors.

