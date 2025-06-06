---
hide:
    - toc
---

# Chapter 16: `torch.storage`, `torch.memory_format`, and Low-Level Tensor Memory

> “Because knowing how your tensors are stored can save you time, memory, and sanity.”

---

## 16.1 What is `torch.storage`?

At its core, every PyTorch Tensor is a **view into a Storage object** — a 1D, contiguous block of memory that holds all the actual data.

### ➤ Access the underlying storage:

```python
x = torch.tensor([[1, 2], [3, 4]])
x_storage = x.storage()
print(list(x_storage))  # Output: [1, 2, 3, 4]
```
x[0][1] accesses the second element, but it’s ultimately referencing an index inside this flat storage.

This abstraction is mostly invisible in modern PyTorch, but helpful for:

- Debugging memory errors

- Understanding views, slicing, and contiguity

- Saving custom binary formats

---

## 16.2 Tensor Views and Storage Sharing

Tensors created from one another may share storage, meaning changes in one affect the other:

```python
a = torch.tensor([1, 2, 3, 4])
b = a.view(2, 2)
b[0][0] = 99
print(a)  # tensor([99, 2, 3, 4]) — same storage!
```
> ✅ If you want independent memory, use .clone():
```python
c = a.clone()
```

---

## 16.3 Contiguity and .is_contiguous()

Contiguous memory layout = row-major (C-style ordering).

```python
x = torch.randn(2, 3)
x_T = x.t()
print(x_T.is_contiguous())  # False — transposing breaks contiguity
```
### ➤ Make tensor contiguous again:
```python
x_T_contig = x_T.contiguous()
```
> ⚠️ Functions like .view() only work on contiguous tensors. <br>
✅ Use .reshape() as a safer alternative.

---

## 16.4 memory_format — Controlling Layout (e.g. NHWC vs NCHW)

In deep learning, layout matters — especially for performance on GPUs or specialized hardware.

### Common formats:
- torch.contiguous_format → NCHW (batch, channel, height, width)

- orch.channels_last → NHWC (optimized for conv2d on GPU)

### ➤ Convert format:
```python
x = torch.randn(1, 3, 224, 224)  # NCHW by default
x_cl = x.to(memory_format=torch.channels_last)
```
### ➤ Check format:
```python
x.is_contiguous(memory_format=torch.channels_last)
```
> ✅ channels_last format boosts performance for 2D convolutions on modern GPUs (A100, RTX, etc.)

---

## 16.5 Saving and Loading Tensors

### ➤ Save raw tensor (uses underlying storage):
```python
torch.save(tensor, 'tensor.pt')
tensor_loaded = torch.load('tensor.pt')
```

### ➤ Write your own binary format:
```python
with open('my_tensor.bin', 'wb') as f:
    f.write(tensor.numpy().tobytes())
```

### ➤ Load from raw bytes:
```python
tensor_from_bin = torch.frombuffer(open('my_tensor.bin', 'rb').read(), dtype=torch.float32)
```
>  Useful for embedded systems, device-to-device transfer, legacy platforms.

---

## 16.6 When Does This Matter?

|Use Case	                    |Why It Matters                         |
|-------------------------------|---------------------------------------|
|Custom tensor manipulation	    |Avoid unintended memory sharing        |
|Model performance (conv2d)	    |Layout format can impact speed         |
|Multi-threading or slicing	    |Views can lead to hidden memory bugs   |
|Saving large datasets	        |Storage-level access may be needed     |
|Deployment to accelerators	    |Requires channels_last layout          |


---

## ✅ 16.7 Summary

|Concept	                |Purpose                                    |
|---------------------------|-------------------------------------------|
|`tensor.storage()`	        |Inspect or share memory manually           |
|`.clone()	`                |Create independent memory copy             |
|`.is_contiguous()`	        |Check if memory is sequential              |
|`.contiguous()`	            |Fix layout to use `.view()` safely           |
|`memory_format`	            |Optimize layout for conv ops (NCHW/NHWC)   |

- Most users won’t need to use `.storage()` directly — but for debugging **weird behavior**, it’s a **lifesaver**

- For performance optimization (especially with CNNs), switching to `channels_last` is one of the **easiest wins**

- Memory format awareness becomes **essential** when building **custom ops**, exporting to **ONNX/TensorRT**, or **scaling** **models**
