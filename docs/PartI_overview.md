---
hide:
    - toc
---

# Part I: Getting Started with `torch`

Welcome to the foundational core of the PyTorch Builder’s Companion Book.

This part introduces the **torch** module—the backbone of the PyTorch ecosystem. Whether you're a beginner eager to understand tensors or an advanced developer optimizing GPU workloads, Part I grounds you in the essential concepts and setup steps required to use PyTorch effectively and confidently.

---

## Chapter 1: What is `torch` and Why Does It Matter?

This chapter lays the philosophical and technical foundation for everything that follows.

- **torch** is not just another module—it's the **core engine** powering every other component in PyTorch.
- We explore its place in the PyTorch ecosystem:  
  - `torch` (core tensor ops),  
  - `torch.nn` (neural network abstractions),  
  - `torch.autograd` (differentiation),  
  - `torchvision/torchaudio/torchtext` (domain-specific wrappers).
- The **torch.Tensor** is introduced as the primary building block—powerful, GPU-ready, and autograd-compatible.
- Key takeaway: To truly master PyTorch, you need to understand torch—not just use it, but *leverage it* for custom layers, debugging, and innovation.

---

## Chapter 2: Installation & Setup

Before building models, you need a proper PyTorch environment.

- Walkthrough of the **installation process** using `pip` or `conda` with CPU or CUDA support.
- Emphasis on using **virtual environments** to avoid polluting your global Python setup.
- Quick verification steps:
  - Importing torch
  - Creating random tensors
  - Checking GPU availability
- Intro to your first mini **Tensor Playground**—creating, operating, and moving tensors to CUDA if available.
- Tips on **CPU vs GPU** behavior, plus troubleshooting advice (e.g., `nvidia-smi`, Python version compatibility).

By the end of this chapter, you’ll be running tensors on your device and preparing for deeper exploration.

---

## Chapter 3: Tensor Fundamentals

This chapter goes deeper into the anatomy of tensors.

- **What is a tensor?** From scalars (0D) to multi-dimensional batches of data (4D+).
- Comparison with NumPy:
  - API similarity
  - GPU support
  - Autograd capability
- Tensor interoperability with NumPy (shared memory warning!).
- Introduction to **device management**:
  - Sending tensors to `'cpu'` or `'cuda'`
  - Choosing a device dynamically
- Use-case comparisons (CPU vs GPU) and why your tensor placement directly impacts performance.

Mastering tensors means mastering data representation, device flow, and performance—skills critical for both beginners and pros.

---

## Summary of Part I

| Concept                | Key Insight                                                                 |
|------------------------|------------------------------------------------------------------------------|
| `torch` Module         | The core engine behind all PyTorch operations                                |
| `torch.Tensor`         | Foundation for all data, models, and gradients in PyTorch                    |
| Installation & Setup   | Prepare clean environments with optional CUDA support                        |
| Device Management      | Know how to work with CPU vs GPU to optimize performance                     |
| Tensor Playground      | Hands-on intro to tensor creation, manipulation, and hardware acceleration   |

Whether you’re debugging deep models or building your first neural net, **Part I arms you with essential tools** to begin your PyTorch journey on solid ground.

→ Next up: A deeper dive into the `torch.Tensor` API.

---

