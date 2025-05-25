# Chapter 12: `torch.nn.functional` and Activation Math

> “When you don’t need a layer, you just need a function.”

---

## 12.1 What is `torch.nn.functional`?

While `torch.nn` contains **layer classes** like `nn.ReLU`, `nn.Linear`, etc., the `torch.nn.functional` module gives you **stateless functional** versions.

| Type                | Example from `torch.nn.functional`         |
|---------------------|---------------------------------------------|
| Activation functions | `F.relu`, `F.sigmoid`, `F.softmax`         |
| Loss functions       | `F.cross_entropy`, `F.mse_loss`            |
| Convolutional ops    | `F.conv2d`, `F.max_pool2d`                 |
| Normalization        | `F.batch_norm`, `F.layer_norm`             |
| Utility transforms   | `F.pad`, `F.interpolate`, `F.one_hot`      |

> **Stateless** means you must pass **all arguments explicitly** — no hidden weights or buffers.

---

## 12.2 Common Activation Functions

```python
import torch.nn.functional as F
x = torch.tensor([-1.0, 0.0, 1.0])
```
### ➤ ReLU
```python
F.relu(x)  # tensor([0., 0., 1.])
```
### ➤ Sigmoid
```python
F.sigmoid(x)  # tensor([0.2689, 0.5000, 0.7311])
```

### ➤ Tanh
```python
F.tanh(x)  # tensor([-0.7616,  0.0000,  0.7616])
```

### ➤ Softmax
```python
logits = torch.tensor([2.0, 1.0, 0.1])
F.softmax(logits, dim=0)  # Probabilities that sum to 1
```
> ✅ Always specify `dim` with `softmax`.

---

## 12.3 Loss Functions in Functional

Loss functions in F behave like their nn counterparts — but are **stateless**.

### ➤ Cross-Entropy Loss
```python
logits = torch.tensor([[2.0, 1.0, 0.1]])
targets = torch.tensor([0])
loss = F.cross_entropy(logits, targets)
```
> F.cross_entropy() = log_softmax + nll_loss internally.
So pass raw logits, not softmaxed outputs.

### ➤ MSE Loss
```python
pred = torch.tensor([0.5, 0.7])
target = torch.tensor([1.0, 0.0])
F.mse_loss(pred, target)
```

### ➤ Binary Cross-Entropy
```python
F.binary_cross_entropy(torch.sigmoid(pred), target)
```

---

##  12.4 `functional` vs `nn.Module`
|Situation	                        |Use                        |
|-----------------------------------|---------------------------|
|You need modularity	            |`nn.ReLU()`, `nn.Linear()`     |
|You want fine-grained control	    |`F.relu()`, `F.linear()`       |
|Inside `forward()` method	        |Prefer` F.*` for functions   |
|Initializing outside model	        |Use `nn.Module` versions     |

### Example:
```python
# With Module
self.relu = nn.ReLU()
x = self.relu(x)

# With Functional
import torch.nn.functional as F
x = F.relu(x)
```
> Inside `forward()`, many developers prefer `F.*` to keep the model class minimal and explicit.

## 12.5 Functional Layers (Linear, Conv, etc.)

### Functional Linear Layer:
```python
weight = torch.randn(10, 5)
bias = torch.randn(10)
x = torch.randn(1, 5)
F.linear(x, weight, bias)
```
> You're responsible for managing parameters manually.
Useful for:
- Writing custom layers
- Doing meta-learning
- Implementing custom architectures

---

## 12.6 One-Hot Encoding
```python
labels = torch.tensor([0, 2])
F.one_hot(labels, num_classes=3).float()
```
> Perfect for manual cross-entropy implementations or label smoothing.

## 12.7 Other Useful Functions

### ➤ Padding:
```python
F.pad(x, pad=(1, 1), mode='constant', value=0)
```
### ➤ Upsampling / Interpolation:
```python
F.interpolate(image, scale_factor=2, mode='bilinear')
```

---

##  12.8 Caution: Don’t Mix Modules with Functionals Blindly

Mixing `nn.CrossEntropyLoss()` with `F.softmax()`? ❌ Bad idea.

`nn.CrossEntropyLoss()` expects raw logits. Passing softmaxed values will double-softmax your output and lead to training instability.

---

## ✅ 12.9 Summary
|Category	            |Functional API Examples        |
|-----------------------|-------------------------------|
|Activations	        |F.relu, F.softmax, F.tanh      |
|Loss Functions	        |F.cross_entropy, F.mse_loss    |
|Layer Ops	            |F.linear, F.conv2d, F.pad      |

- torch.nn.functional is stateless and explicit.
- Great for flexibility, custom layers, or experimental architectures.
- Use with care — you must manage shapes, devices, and parameters manually.

