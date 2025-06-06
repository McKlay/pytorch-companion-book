---
hide:
    - toc
---

# Chapter 6: Random Sampling and Seeding

> “To master the chaos, you must first control the dice.”

---

## 🎲 6.1 Why Randomness Matters in Deep Learning

Randomness shows up everywhere in machine learning:

- Weight initialization  
- Dropout layers  
- Data shuffling  
- Augmentation  
- Mini-batch selection  
- Sampling in generative models (e.g., GANs)

And PyTorch gives you **robust control** over all of it.

---

## 6.2 Common Random Tensor Generators

PyTorch provides several methods to generate random numbers:

### ➤ `torch.rand(*sizes)`
Returns values ∈ [0, 1), uniform distribution.

```python
torch.rand(2, 3)
```

### ➤ `torch.randn(*sizes)`
Standard normal distribution (mean=0, std=1)
```python
torch.randn(3, 3)
```

### ➤ `torch.randint(low, high, size)`
Uniform integer values from low (inclusive) to high (exclusive)
```python 
torch.randint(0, 10, (2, 2))
```

### ➤ `torch.randperm(n)`
Random permutation of integers from 0 to n-1
```python
torch.randperm(5)  # e.g., tensor([3, 0, 4, 2, 1])
```

### ➤ `torch.bernoulli(probs)`
Samples 0 or 1 based on given probabilities (used for dropout)
```python
probs = torch.tensor([0.1, 0.5, 0.9])
torch.bernoulli(probs)
```

## 6.3 How to Set Seeds for Reproducibility

Randomness is useful — until you need your results to be repeatable.

### Global seed setter:
```python
torch.manual_seed(42)
```
This affects:

- `torch.rand`, `torch.randn`, `torch.randint`

- Initial weights in models

- Dropout patterns

If you're using CUDA:
```python
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
```

## 6.4 `torch.Generator` – When You Want Control Per Operation

Sometimes you want repeatable randomness without resetting the global seed. That’s where `torch.Generator` comes in.

### Example:
```python
g = torch.Generator()
g.manual_seed(123)
x1 = torch.rand(2, 2, generator=g)
x2 = torch.rand(2, 2, generator=g)  # Continues from the same stream
```
This is especially useful when:

- You’re writing tests

- You want independent RNG streams

- You’re managing parallel data loading or multiprocessing

## 6.5 Beware: Randomness in Parallel Training

If you’re training across:

- Multiple GPUs

- Multiple processes (e.g., `DataLoader` with `num_workers > 0`)

- Multiple epochs with shuffling

Then you must control randomness carefully or you’ll get:

- Nondeterministic results

- Flaky training behavior

- Inconsistent evaluation metrics

### To mitigate this:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
And in `DataLoader:`
```python
DataLoader(..., worker_init_fn=seed_worker)
```
Where `seed_worker` reseeds per worker.

##  6.6 Summary

- PyTorch offers `rand`, `randn`, `randint`, and `randperm` for common randomness.

- Use `torch.manual_seed()` to control global RNG for reproducibility.

- Use `torch.Generator` for isolated, repeatable randomness.

- Beware: parallelism and dropout can make reproducibility tricky without proper seeding.