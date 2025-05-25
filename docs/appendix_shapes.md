# Appendix A: Tensor Shapes Cheat Sheet

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