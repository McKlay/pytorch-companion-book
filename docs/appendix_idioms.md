---
hide:
    - toc
---

# Appendix B: PyTorch Idioms and Gotchas

> “Read this before your model does something stupid.”

## Idioms

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

## ⚠️ Gotchas

| Gotcha                          | What Goes Wrong                      | Fix                                   |
|----------------------------------|--------------------------------------|----------------------------------------|
| Using `.data` for detaching      | Breaks autograd silently             | Use `.detach()`                        |
| Calling `.numpy()` on CUDA tensor| Immediate crash                      | Move to CPU first                      |
| In-place ops like `+=`, `add_()` | May break gradients                  | Use regular ops (`x = x + y`)          |
| Forgetting `.train()` / `.eval()`| BatchNorm/Dropout misbehave          | Switch mode explicitly                 |
| Wrong loss input types           | Float tensor vs. int labels          | Match dtypes (`.float()`)              |
| Not zeroing `.grad` before `.backward()` | Gradients accumulate      | Call `optimizer.zero_grad()`          |

---