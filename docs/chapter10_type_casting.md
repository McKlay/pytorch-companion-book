# Chapter 10: Type Conversions and Casting

> “Wrong dtype, wrong device — game over.”

---

## 🔄 10.1 Why Casting Matters

Tensor operations require matching **types** and **devices**. If not, you’ll face runtime errors like:
```python
RuntimeError: expected scalar type Float but found Long
```

Or worse — **silently incorrect results**.

So casting properly is not just about avoiding bugs — it’s about making your models work as intended.

---

## 10.2 Basic Casting Methods

### ➤ Convert to Float, Long, Int, Bool

```python
x = torch.tensor([1, 0, 1])
x.float()      # torch.float32
x.long()       # torch.int64
x.int()        # torch.int32
x.bool()       # torch.bool
```
These are shorthand wrappers around .to(dtype=...).

---

## 10.3 .to() — The Multipurpose Transformer

`.to()` can change:
- The dtype
- The device
- Or both at once

```python
x = x.to(torch.float32)
x = x.to('cuda')
x = x.to(torch.float16, device='cuda')
```
> ✅ Best practice: use .to() for cross-device + dtype-safe conversions in pipelines.

---

## 10.4 .type() — Less Flexible, Still Useful

```python
x = x.type(torch.FloatTensor)
```
> But it doesn’t support device changes, so `.to()` is preferred in most modern PyTorch code.

---

## 10.5 Matching Types in Ops

This error is extremely common when dealing with losses or metrics:
```python
preds = torch.tensor([0.6, 0.2, 0.8])         # float32
labels = torch.tensor([1, 0, 1])              # int64
loss = torch.nn.BCELoss()
loss(preds, labels)   # ❌ Throws error
```
### ✅ Fix:
```python
labels = labels.float()
```
> Most loss functions expect both inputs to be `float32`, not integers.

---

##  10.6 One-Hot Encoding Pitfall

When using one-hot encodings, make sure your tensors are in `float` (or `bool` in some masking scenarios):
```python
y = torch.tensor([0, 2])
one_hot = torch.nn.functional.one_hot(y, num_classes=3).float()
```
> If you pass long-typed one-hot vectors into models, expect weird gradients or silent failures.

## 10.7 Converting Between Devices Safely
Avoid this:
```python
x.cuda()
```
Use this:
```python
x.to('cuda')  # Safer, cleaner, can be combined with dtype
```
To stay cross-platform, define your device once:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

---

## 🚫 10.8 Dangerous Casting Practices
|Practice	                |Why It's Bad	            |Fix                    |
|---------------------------|---------------------------|-----------------------|
|Using `.data`	            |Bypasses autograd	        |Use `.detach()`          |
|Forgetting `.float()`	    |Breaks loss functions	    |Always match model dtype|
|Manual `.cuda()` calls	    |Breaks portability	        |Use `.to(device)`        |
|Mixing float64 + float32	|Silently slows ops	        |Use consistent dtype   |
|                           |                           |                       |

---

## 10.9 Summary
- Use `.float()`, `.long()`, `.bool()` for quick casting.

- Use .`to(dtype, device)` to control both type and placement.

- Prefer `.to()` over `.type()` and never use `.data`.

- Always check input/output types before applying loss or activation functions.

- Define your device once and pass it around to keep your code portable.