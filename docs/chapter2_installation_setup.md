---
hide:
    - toc
---

# Chapter 2: Installation & Setup

> “A neural net’s journey begins with a single tensor.”

---

## 2.1 Installing PyTorch the Right Way

## ✅ Step 1: Visit the Official Installer Page

Go to: [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

You’ll see a selector for:

- PyTorch Build (Stable / Preview)
- Your OS (Linux, Mac, Windows)
- Package Manager (`pip`, `conda`)
- Language (`Python`, `C++`)
- Compute Platform (`CPU`, `CUDA 11.8`, `CUDA 12.x`, etc.)

## Recommendation

For most ML developers (especially if you’re using a GPU):

```bash
# For pip + CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If you don’t have a GPU, install CPU-only:

```bash
pip3 install torch torchvision torchaudio
```
> ⚠️ Heads-up: Make sure your Python version is 3.8–3.12. PyTorch isn’t too happy with older versions.

## 2.2 Virtual Environment Setup (Highly Recommended)

To avoid breaking other Python projects:
```bash
python -m venv torch_env
source torch_env/bin/activate  # On Windows: torch_env\Scripts\activate
```
Then install PyTorch inside that environment.

## 2.3 Verify the Installation

Let’s test it right away:
```bash
import torch
x = torch.rand(3, 3)
print("Tensor:\n", x)
print("Is CUDA available?", torch.cuda.is_available())
```
You should see a 3×3 matrix of random values and a True/False flag about CUDA.

##  2.4 Your First Tensor Playground

Let’s walk through a few core operations to prove it works — and start building intuition.

➤ Create a Tensor
```bash
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
```
➤ Do Some Math
```bash
b = torch.ones_like(a)
c = a + b
print("Addition:\n", c)
```
➤ Matrix Multiplication
```bash
d = torch.matmul(a, c.T)  # Transpose and multiply
print("Matrix product:\n", d)
```
➤ Move to GPU (if available)
```bash
if torch.cuda.is_available():
    a = a.to("cuda")
    print("Tensor on GPU:", a)
```

## 2.5 CPU vs CUDA: Why It Matters

|Operation	            |CPU	                        |CUDA (GPU)                 |
|-----------------------|-------------------------------|---------------------------|
|Matrix multiplication	|Slower for large matrices	    |Highly optimized           |
|Memory access	        |Direct system memory	        |VRAM on GPU                |
|Use-case	            |Lightweight ML / debugging	    |Training large models      |

TL;DR: Use CUDA if available. It’s fast. Like, ridiculously fast.

## 2.6 Troubleshooting Tips

❌ “torch not found”
Make sure:

 - You activated the correct environment

 - You installed it in the right Python version

❌ CUDA not available
 - Check your driver:
```bash
nvidia-smi
```
> Make sure your CUDA toolkit matches the version you selected during install.

##  2.7 Quick Recap
By now, you should:

✅ Have PyTorch installed in a clean environment

✅ Know how to write and run basic tensor operations

✅ Understand the role of CUDA and how to check if it’s working