---
hide:
    - toc
---

# Chapter 9: Autograd and Differentiation

> “If tensors are the muscles, autograd is the nervous system.”

---

## 9.1 What Is Autograd?

Autograd is PyTorch’s **automatic differentiation engine**.  
It builds a computation graph behind the scenes as you operate on tensors with `requires_grad=True`. When you call `.backward()`, it traces back through that graph to compute gradients.

> This is what powers training in PyTorch — from basic logistic regression to massive transformers.

---

## 9.2 Enabling Gradient Tracking

To start tracking gradients:

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
```
> Now, any operation on `x` will be recorded:
```python
y = x ** 2 + 3
z = y.sum()
z.backward()
print(x.grad)  # Output: tensor([4., 6.])
```
> Here, ∂z/∂x = 2x → [2×2, 2×3] = [4., 6.]

## 9.3 Calling `.backward()`

Once you have a scalar result (like loss), call:
```python
loss = model(input).sum()
loss.backward()
```
PyTorch will:

- Walk backward through the computation graph

- Calculate gradients for every tensor with `requires_grad=True`

- Store gradients in the `.grad` attribute

## 9.4 Stopping Gradient Tracking

When you want to freeze parts of the model (e.g., during evaluation or feature extraction), use:

Method 1: with `torch.no_grad()`
```python
with torch.no_grad():
    y = model(x)
```
Method 2: `detach()`
```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
z = y.detach()  # z does not track gradients
```

## 9.5 Checking the Computation Graph

You can inspect how PyTorch built the graph:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x * x
print(y.grad_fn)  # Output: <MulBackward0>
```
> Every operation creates a `Function` object like `AddBackward0`, `MulBackward0`, etc.
This is how PyTorch knows how to differentiate each step.

## 9.6 `torch.autograd.Function`: Custom Gradients

If you want to write your own forward/backward logic (like building a custom layer or operator):
```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input
```

Use it like:
```python
x = torch.tensor([3.0], requires_grad=True)
y = Square.apply(x)
y.backward()
print(x.grad)  # Should be 6.0
```
> 📌 Advanced, but useful for low-level ops and optimization research.

## 9.7 Common Mistakes

|Mistake	                                |Fix                                                    |
|-------------------------------------------|-------------------------------------------------------|
|Calling `.backward()` on non-scalar	        |Pass a gradient argument: `z.backward(torch.ones_like(z))` |
|Using `.data` instead of `.detach()`	        |Use `.detach()` — `.data` is risky                 |
|In-place ops corrupting graph	            |Avoid `x += ...` with autograd-tied tensors            |
|Forgetting .`zero_()` on `.grad`	            |Always zero gradients before `.backward()`         |

```python
optimizer.zero_grad()  # Or model.zero_grad()
loss.backward()
optimizer.step()
```

##  9.8 Gradient Accumulation

By default, gradients accumulate:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 2
y.backward()
y.backward()
print(x.grad)  # Will be 4 + 4 = 8
```
Use x.grad.zero_() or optimizer.zero_grad() to prevent this.

---

## 9.9 Summary

- `requires_grad=True` enables gradient tracking for a tensor.

- `.backward()` triggers backpropagation from a scalar output.

- Gradients are stored in `.grad`.

- Use `no_grad()` or `detach()` to stop tracking.

- Autograd builds a dynamic graph as you run — no static declarations.


