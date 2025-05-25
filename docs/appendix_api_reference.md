# Appendices

---

## 🔢 Appendix A: Tensor Shapes Cheat Sheet

> “Because debugging starts with dimensions.”

| Task / Layer               | Expected Shape                  |
|----------------------------|----------------------------------|
| Single image (grayscale)   | `[1, H, W]`                      |
| Single image (RGB)         | `[3, H, W]`                      |
| Batch of grayscale images  | `[B, 1, H, W]`                   |
| Batch of RGB images        | `[B, 3, H, W]`                   |
| Fully connected input      | `[B, features]`                 |
| LSTM input                 | `[seq_len, batch, input_size]` or `[B, T, F]` (`batch_first=True`) |
| Transformer input          | `[B, seq_len]` or `[B, seq_len, emb_dim]` |
| `nn.Embedding` input       | `[B, T]` (int64 tokens) → Output: `[B, T, emb_dim]` |
| CNN output to FC layer     | `[B, C, H, W]` → `x.view(B, -1)` |
| Classification output      | `[B, num_classes]`              |
| Regression output          | `[B, 1]` or `[B]`                |

> Always `print(.shape)` at each step of your model to avoid dimensional disasters.

---

## 💡 Appendix B: PyTorch Idioms and Gotchas

> “Read this before your model does something stupid.”

### ✅ Idioms

| Goal                       | Idiomatic PyTorch Code                                |
|----------------------------|--------------------------------------------------------|
| Move everything to GPU     | `x = x.to(device); model = model.to(device)`          |
| Get model predictions      | `with torch.no_grad(): output = model(x)`             |
| Detach and convert to NumPy| `x.detach().cpu().numpy()`                            |
| One-hot encode             | `F.one_hot(t.long(), num_classes).float()`            |
| Check for NaNs             | `torch.isnan(x).any()`                                |
| Log training stats         | `writer.add_scalar('loss', val, step)`                |
| Save model                 | `torch.save(model.state_dict(), 'model.pt')`          |
| Load model                 | `model.load_state_dict(torch.load(...))`              |

### ⚠️ Gotchas

| Gotcha                          | What Goes Wrong                      | Fix                                   |
|----------------------------------|--------------------------------------|----------------------------------------|
| Using `.data` for detaching      | Breaks autograd silently             | Use `.detach()`                        |
| Calling `.numpy()` on CUDA tensor| Immediate crash                      | Move to CPU first                      |
| In-place ops like `+=`, `add_()` | May break gradients                  | Use regular ops (`x = x + y`)          |
| Forgetting `.train()` / `.eval()`| BatchNorm/Dropout misbehave          | Switch mode explicitly                 |
| Wrong loss input types           | Float tensor vs. int labels          | Match dtypes (`.float()`)              |
| Not zeroing `.grad` before `.backward()` | Gradients accumulate      | Call `optimizer.zero_grad()`          |

---

## 🗂 Appendix C: Full `torch` API Reference Crosswalk

> “All the power. One list.”

| Module                       | Link                                                                                   | Description                                  |
|-----------------------------|----------------------------------------------------------------------------------------|----------------------------------------------|
| `torch.Tensor`              | https://pytorch.org/docs/stable/tensors.html                                          | Core data structure                          |
| `torch.nn`                  | https://pytorch.org/docs/stable/nn.html                                               | Layers, loss functions, model building       |
| `torch.nn.functional`       | https://pytorch.org/docs/stable/nn.functional.html                                    | Stateless functional ops                     |
| `torch.autograd`            | https://pytorch.org/docs/stable/autograd.html                                         | Gradient tracking, custom ops                |
| `torch.cuda`                | https://pytorch.org/docs/stable/cuda.html                                             | GPU support, memory info                     |
| `torch.utils.data`          | https://pytorch.org/docs/stable/data.html                                             | Dataset, DataLoader, Samplers                |
| `torch.special`             | https://pytorch.org/docs/stable/special.html                                          | Advanced math functions (gamma, digamma, etc.) |
| `torch.fft`                 | https://pytorch.org/docs/stable/fft.html                                              | Frequency transforms (fft, rfft, fft2, etc.) |
| `torch.linalg`              | https://pytorch.org/docs/stable/linalg.html                                           | Modern linear algebra tools                  |
| `torch.profiler`            | https://pytorch.org/docs/stable/profiler.html                                         | CPU/GPU performance profiling                |
| `torch.onnx`                | https://pytorch.org/docs/stable/onnx.html                                             | Exporting to ONNX format                     |
| `torch.jit`                 | https://pytorch.org/docs/stable/jit.html                                              | Scripting & tracing models for speed/export  |
| `torchvision.transforms`    | https://pytorch.org/vision/stable/transforms.html                                     | Image preprocessing                          |
| `torch.utils.tensorboard`   | https://pytorch.org/docs/stable/tensorboard.html                                      | Training visualizations                      |
