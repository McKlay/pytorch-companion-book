# Chapter 15: `torch.utils`

> “Without utils, it’s just you and a for-loop in the wilderness.”

---

## 15.1 What is `torch.utils`?

`torch.utils` is a collection of **essential utilities** that make PyTorch practical for real-world training:

- `data`: Dataset & DataLoader interface  
- `tensorboard`: Visualize training progress  
- `checkpoint`: Save & resume models / reduce memory  
- `cpp_extension`, `throughput_benchmark`, etc.

> We’ll focus on the most important parts you’ll use **almost daily**.

---

## 15.2 `torch.utils.data` — Custom Datasets and DataLoaders

This submodule is the **backbone of PyTorch’s training loop**.

### ➤ Create a Custom Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
```

### ➤ Use with DataLoader
```python
from torch.utils.data import DataLoader

dataset = MyDataset(torch.randn(100, 3), torch.randint(0, 2, (100,)))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch_data, batch_labels in loader:
    pass  # training loop here
```
> ✅ DataLoader handles batching, shuffling, multiprocessing (num_workers), and pinning memory to improve performance.

---

## 15.3 Built-in Helpers

### ➤ TensorDataset — Wrap tensors directly
```python
from torch.utils.data import TensorDataset

ds = TensorDataset(torch.randn(100, 3), torch.randint(0, 2, (100,)))
```

### ➤ `ConcatDataset`, `Subset`, `RandomSampler`
These let you:

- Combine datasets (`ConcatDataset`)
- Take slices (`Subset`)
- Customize sample orders (`Sampler`, `WeightedRandomSampler`)

---

##  15.4 `torch.utils.tensorboard` — Visualize Your Training

### ➤ Track loss, accuracy, gradients:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for epoch in range(10):
    train_loss = 0.5 * (10 - epoch)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_histogram('Weights/layer1', model.layer1.weight, epoch)

writer.flush()
```
Then launch with:
```bash
tensorboard --logdir=runs/
```
> Great for comparing experiments, visualizing embeddings, and debugging weird training plateaus.

---

## 5.5 `torch.utils.checkpoint` — Save Memory with Recompute

Use this to trade compute for memory in huge models like Transformers or ResNets.

### ➤ Wrap part of the forward pass:
```python
from torch.utils.checkpoint import checkpoint

def custom_forward(*inputs):
    return model(*inputs)

output = checkpoint(custom_forward, x)
```
- PyTorch will discard intermediate tensors during forward()

- Recompute them during .backward() to save GPU memory

---

## 15.6 Other Utilities

### ➤ `throughput_benchmark`
Used to test model speed under realistic loads.

### ➤ `cpp_extension`
Compile and call custom CUDA/C++ kernels from Python — used in:
- Detectron2
- Hugging Face Transformers
- Other low-level optimization libraries

> Very advanced — only needed if you’re building custom operators or native extensions.

---

## 15.7 Real-World Workflow Using `torch.utils`
|Task	                    |Tool Used                      |
|---------------------------|-------------------------------|
|Batch loading	            |`DataLoader`                   |
|Custom data pipeline	    |`Dataset` subclass             |
|Visual logging	            |`tensorboard.SummaryWriter`    |
|Save VRAM in deep nets	    |`checkpoint()`                 |
|Combine multiple datasets	|`ConcatDataset`, `Subset`      |

> ✅ These are the unsung heroes of PyTorch — not mathematical, but without them, every model would be a pain to train, debug, or scale.

---

## ✅ 15.8 Summary

|Submodule	            |Use Case                                   |
|-----------------------|-------------------------------------------|
|`data`	                |Datasets, DataLoaders, batching            |
|`tensorboard`	        |Visualize metrics, histograms, images      |
|`checkpoint`	            |Memory-efficient training                  |
|`cpp_extension`	        |Write custom kernels (advanced)            |


- `torch.utils` turns raw PyTorch into a full-blown ML framework

- You’ll use `data` and `tensorboard` in every project

- Use `checkpoint` when running out of VRAM or fitting giant nets