---
hide:
    - toc
---

# Part III: Specialized Modules in `torch`

This part of the book takes you beyond tensors and core math — into the **specialized submodules** of `torch` that unlock powerful capabilities for scientific computing, custom modeling, signal processing, and training workflows.

Whether you're building large neural nets, performing linear algebra, or squeezing performance out of low-level memory layout — this section equips you with the tools.

---

## Chapter 11: `torch.linalg`

This module brings modern, NumPy-style linear algebra into PyTorch:

- Matrix inversion: `torch.linalg.inv()` (use `solve()` instead when possible)
- Solving systems: `torch.linalg.solve(A, b)` for Ax = b
- Determinants, ranks, norms, and condition numbers
- Eigenvalues and SVD: `eig`, `svd`, `qr`, `lu`
- GPU support, autograd compatibility, and batched ops
- Ideal for PCA, model diagnostics, and physics-inspired ML

*Pro tip*: Prefer `solve()` over `inv()` for better numerical stability.

---

## Chapter 12: `torch.nn.functional`

Stateless functions for deep learning components:

- Activations: `F.relu`, `F.sigmoid`, `F.softmax`, etc.
- Losses: `F.cross_entropy`, `F.mse_loss`, `F.binary_cross_entropy`
- Functional layers: `F.linear`, `F.conv2d`, `F.pad`, `F.interpolate`
- Use inside `forward()` methods for explicit control
- Essential for custom layers, dynamic architectures, and meta-learning

*Reminder*: Never pass softmaxed logits into `F.cross_entropy()` — it expects raw logits.

---

## Chapter 13: `torch.special`

Advanced mathematical functions for specialized models:

- `expit`, `erf`, `gamma`, `lgamma`, `digamma`, `xlogy`, `i0`, etc.
- Useful in: statistical distributions, variational inference, RL, and probabilistic modeling
- Inspired by SciPy's `scipy.special`, but supports autograd and GPU
- Handles numerical stability (e.g., `xlogy(0, 0)` safely)

*Use case*: KL divergence, entropy, Bayesian models, and scientific ML.

---

## Chapter 14: `torch.fft`

Time-to-frequency domain transformations:

- 1D FFT: `fft`, `ifft`, `rfft`, `irfft`
- 2D/ND FFT: `fft2`, `ifft2`, `fftn`
- Real-world use: audio signal analysis, image filtering, denoising, spectral CNNs
- All FFT outputs are complex tensors: `.real`, `.imag`, `.abs()`, `.angle()`

*Example*: Low-pass filtering noisy signals by zeroing high frequencies in FFT space.

---

## 🛠 Chapter 15: `torch.utils`

Utilities that make PyTorch practical and scalable:

- `torch.utils.data`: Datasets and DataLoaders for efficient training
- `tensorboard`: Visualize loss curves, histograms, metrics
- `checkpoint`: Save memory by recomputing during backprop (useful in ResNets, Transformers)
- Others: `ConcatDataset`, `Subset`, `Sampler`, `cpp_extension`, `throughput_benchmark`

💡 *Reminder*: You’ll use `DataLoader` and `SummaryWriter` in almost every real-world project.

---

## Chapter 16: Low-Level Tensor Memory & Storage

Understand the guts of how tensors are stored and accessed:

- `.storage()`: Access the underlying flat memory
- `.is_contiguous()`: Check for sequential memory layout
- `.contiguous()`: Required for `.view()` and some backends
- `memory_format`: Control layout for CNNs (NCHW vs NHWC)
- `torch.save()` / `torch.load()` for raw tensor serialization

*Tip*: Use `channels_last` layout for faster 2D convolutions on modern GPUs.

---

## Summary of Part III

| Chapter | Submodule            | What You Gain                                                       |
|---------|----------------------|---------------------------------------------------------------------|
| 11      | `torch.linalg`       | Modern linear algebra with GPU, autograd, and batching             |
| 12      | `torch.nn.functional`| Stateless, flexible layer & loss functions                         |
| 13      | `torch.special`      | Advanced math for scientific, probabilistic, and stable modeling   |
| 14      | `torch.fft`          | Frequency domain processing for signals and images                 |
| 15      | `torch.utils`        | Dataset loading, visualization, memory-saving, and training tools  |
| 16      | `.storage`, `memory_format` | Low-level control over memory, views, and performance tuning    |

**This part unlocks the full computational power of PyTorch**, from mathematics to model utilities. Whether you're building custom layers, optimizing convolutions, or working with scientific data — these tools are how you scale.

→ Next up: Part IV — Using CUDA, Mixed Precision, and Deployment Strategies.

---
