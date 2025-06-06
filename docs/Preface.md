---
hide:
  - toc
---

### Why This Book Exists

Most people meet PyTorch through beginner tutorials that work — until they don’t. You copy a few lines of code, tweak a layer or a tensor, and everything seems fine… until you hit a cryptic shape mismatch, an exploding loss, or a silent autograd bug that derails your model.

This book was created to solve that.

The **PyTorch Builder’s Companion Book** is not just a walkthrough of the `torch` API — it’s a builder’s map of PyTorch’s computational engine. I wrote it after months of working directly with `torch.Tensor`, tracing bugs in gradient flows, and digging into the internals of autograd, CUDA, and advanced training setups.

If you’ve ever wondered why your gradients vanish, why `.view()` fails silently, or how tensors actually move between devices — this book is for you.

This is a guide **for those who build things and want PyTorch to feel like a tool, not a black box.**

---

### Who Should Read This

This book is written for:

- **Engineers and AI builders** who want to understand the internals of PyTorch — not just use `nn.Linear` and hope it works.
- **TensorFlow or NumPy users** transitioning to PyTorch who want deep API fluency.
- **Graduate students or thesis writers** looking to customize training loops or create new architectures.
- **Anyone debugging PyTorch code** who wants clarity around tensors, devices, gradients, or performance.

You don’t need to be a math wizard. But you do need curiosity — and a willingness to follow a tensor from shape to shape, from CPU to GPU, and from forward to backward.

---

### From Tensors to Gradients: How This Book Was Born

This book didn’t begin as a textbook. It began as survival notes.

Notes on why `.detach()` works but `.data` doesn’t. On why `view()` fails after `.permute()`. On how to debug silent failures in custom loss functions. On what exactly happens during backpropagation inside PyTorch.

Eventually, those notes turned into principles. The principles became diagrams and code. And those became chapters.

If you’ve ever tried to freeze part of a pretrained model or troubleshoot why `.backward()` gives `None`, you're not alone. This book is a distillation of that confusion — turned into clarity.

---

### What You’ll Learn (and What You Won’t)

You will learn:

- How `torch.Tensor` works: creation, reshaping, slicing, and memory layout
- The difference between `float32`, `float16`, `bfloat16`, and when to use each
- How autograd builds computation graphs — and how to debug them
- How to use `torch.nn.functional`, `torch.fft`, `torch.linalg`, and more
- Real-world skills: using CUDA, AMP, mixed precision, and multi-GPU training
- How to write PyTorch code that is readable, robust, and fast

You will *not* find:

- Overly abstract explanations with no runnable code
- High-level metaphors that gloss over mechanics
- Tutorials that "just work" without explaining why

This is a book about **what PyTorch really does** — and how to wield it with precision.

---

### How to Read This Book (Even if You’re Just Starting Out)

Each chapter is structured around:

- **Conceptual Insight** – what this torch API does and why it matters  
- **Code Walkthrough** – annotated, working examples with expected outputs  
- **Debugging Tips** – common edge cases and how to fix them  
- **Practice Prompt** – real tasks to try on your own  
- **Use Case Callout** – where this tool fits into real AI workflows

If you're coming from TensorFlow, focus on the PyTorch equivalents. If you're new to deep learning, lean on the diagrams, visual cues, and hands-on exercises.

You don’t need to master everything at once. But by the end, you’ll be able to build, train, debug, and deploy with PyTorch — **not as a user, but as an engineer who understands the framework inside and out.**

---

*Written and maintained by [Clay Mark Sarte](https://www.linkedin.com/in/clay-mark-sarte-283855147/)*
