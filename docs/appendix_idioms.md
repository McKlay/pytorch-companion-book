# Idioms

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