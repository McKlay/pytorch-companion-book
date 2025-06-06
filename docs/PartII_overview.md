---
hide:
    - toc
---

# Part II: `torch` API Deep Dive

Now that you've set up your environment and grasped the essence of `torch`, it’s time to **dive into its core APIs**. This part is where torch transforms from a black box to a precision tool in your hands.

From tensor manipulation to GPU control, autograd magic to precision casting — this section walks through **everything you need to know to operate at the atomic level of PyTorch programming**.

---

## Chapter 4: `torch.Tensor`

The heart of PyTorch. This chapter covers:

- How to create tensors: from lists, with factory methods (`zeros`, `ones`, `full`, etc.), or based on other tensors
- Tensor attributes: `shape`, `dtype`, `device`, and `requires_grad`
- Common operations: arithmetic, logical comparisons, reshaping, slicing
- Tensor reshaping: `view`, `reshape`, `permute`, `squeeze`, `unsqueeze`
- Autograd compatibility: how tensors track operations to support backpropagation
- Pro tricks: `clone()`, `detach()`, and contiguous memory

Understanding `torch.Tensor` is foundational — every model, every dataset, every operation begins here.

---

## Chapter 5: Data Types and Devices

Precision and location matter. This chapter explores:

- `torch.dtype`: float32, float64, float16, int32, bool, and when to use each
- `torch.device`: placing tensors on CPU vs GPU (and how to move them)
- Default dtype control: globally set default types for future tensors
- Mixed precision training: introduction to AMP for faster, memory-efficient training

Mastering `dtype` and `device` ensures that your models run **correctly, fast, and on the right hardware**.

---

## Chapter 6: Random Sampling and Seeding

Randomness is everywhere in ML — but chaos isn’t always good. Learn how to:

- Generate random tensors with `rand`, `randn`, `randint`, `randperm`, and `bernoulli`
- Set global seeds using `torch.manual_seed()` and `torch.cuda.manual_seed_all()`
- Use `torch.Generator` for repeatable randomness without global impact
- Avoid pitfalls in parallel training by seeding workers and managing `cudnn` flags

Control over randomness is key to **reproducibility**, debugging, and experimentation.

---

## Chapter 7: Math Operations

This chapter decodes PyTorch’s math capabilities:

- Elementwise math: `+`, `-`, `*`, `torch.exp`, `torch.log`, etc.
- Reduction ops: `sum`, `mean`, `max`, `argmax`, `prod`
- Matrix operations: `matmul`, `@`, `bmm`, `einsum` for advanced contractions
- Special math: `clamp`, `abs`, `round`, normalization formulas
- In-place ops and performance tips (with warnings for autograd safety)

Knowing how to **express math elegantly and efficiently** is a superpower for model builders.

---

## Chapter 8: Broadcasting and Shape Ops

Shape mismatch errors are a rite of passage. This chapter teaches you how to:

- Understand **broadcasting rules** and how PyTorch aligns mismatched shapes
- Master reshaping tools: `view`, `reshape`, `squeeze`, `unsqueeze`, `permute`, `transpose`
- Optimize memory with `expand()` over `repeat()`
- Apply shape ops to real-world use cases: CNNs, batch processing, and label matching
- Debug shape bugs with confidence

Broadcasting + shape ops = **clean code and fewer runtime nightmares**.

---

## Chapter 9: Autograd and Differentiation

Meet the nervous system of PyTorch:

- `requires_grad=True`: mark tensors for gradient tracking
- `.backward()`: compute gradients through the computation graph
- `.grad`: access the result of differentiation
- `torch.no_grad()` and `.detach()` to freeze parts of your model
- Build custom gradients with `torch.autograd.Function`
- Common mistakes (e.g., in-place ops, forgetting `zero_()`, calling `.backward()` on non-scalars)

This chapter reveals how PyTorch powers **training loops, optimizers, and learning**.

---

## Chapter 10: Type Conversions and Casting

Dtypes and devices must match — or errors ensue.

- Quick casting: `.float()`, `.long()`, `.bool()`
- The mighty `.to()`: safely cast and move tensors
- `.type()` vs `.to()` and why the latter is preferred
- Safe device transitions and best practices for model portability
- Common bugs: mismatched dtypes in loss functions, unsafe use of `.data`, casting silently breaking autograd

Precision matters — and so does **defensive coding** with proper casting.

---

## Summary of Part II

| Chapter | Core Focus                              | Key Skills Gained                                  |
|--------|------------------------------------------|----------------------------------------------------|
| 4      | Tensor Object                            | Creation, reshaping, autograd compatibility        |
| 5      | Data Types and Devices                   | Precision control, CUDA placement, AMP intro       |
| 6      | Randomness and Reproducibility           | Seeding, RNG generators, parallel consistency      |
| 7      | Math Operations                          | Arithmetic, reductions, matrix math, einsum        |
| 8      | Broadcasting and Shape Management        | Shape alignment, expand vs repeat, permute mastery |
| 9      | Autograd and Differentiation             | Gradient tracking, custom gradients, backprop      |
| 10     | Casting and Type Safety                  | .to(), .float(), device-aware conversion           |

This section gives you **full command of torch's deep API layer** — a toolbox you'll use daily for model training, debugging, and performance tuning.

→ Coming next: Part III — Diving into `torch.nn`, model construction, and building blocks of neural networks.

---
