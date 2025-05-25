# Chapter 19: Debugging, Profiling, and Best Practices

> “Where code either gets smarter… or gets you fired.”

Let’s wrap up Part IV with the good stuff: not the fancy models or sexy math, but the tools that make sure your code doesn’t silently ruin your entire experiment while you’re staring at a loss curve wondering what went wrong.

This chapter is your battle-tested field guide for debugging, profiling, and writing PyTorch that doesn’t betray you.

---

## 19.1 Debugging Tensor Values

**First rule of PyTorch debugging:** Check your tensors early and often.

```python
print(tensor.shape)
print(torch.isnan(tensor).any())
print(torch.isinf(tensor).any())
```

### ➤ Check for exploding/vanishing gradients:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```


---


## 19.2 Common Silent Killers

|Bug	                         |Symptom	                                |Fix        |
|--------------------------------|------------------------------------------|--------|
|Using .data	                 |Breaks autograd	                        |Use `.detach()`|
|Mixing CPU and CUDA tensors	 |RuntimeError or silent slowdown	        |Use `.to(device)` consistently|
|Forgetting model.train()	     |Dropout/BatchNorm behaves incorrectly	    |Always use `.train()` / .`eval()`|
|Wrong input shapes	             |Model compiles but outputs garbage	        |Print input/output shapes before layers|
|In-place ops	                 |Loss gets stuck / None gradients	        |Avoid `x += y`; use `x = x + y`|


---

## 19.3 Debug Mode with torch.autograd.set_detect_anomaly

Use this to catch:

- In-place ops that break gradients

- NaNs in backward pass

- Invalid computation graph paths

```python
with torch.autograd.set_detect_anomaly(True):
    loss.backward()
```
> ⚠️ Slightly slower — but worth it during debugging.

---

##  19.4 Profiler for Performance Tuning

Basic usage:
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    run_training_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```
> Shows CPU and GPU time per op — useful for finding bottlenecks.

---

## 19.5 Tracking GPU Memory

```python
print(torch.cuda.memory_summary(device=None, abbreviated=False))
```
Common VRAM hogs:

- Large models without checkpoint()

- Storing intermediate results (forgetting to .detach())

- Retaining computation graphs across batches

---

## 19.6 Tips for Clean, Modular Code

Use a consistent device management strategy:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
model = model.to(device)
```
Structure your project:

- `model.py` — architectures

- `train.py` — training loop

- `utils.py` — reusable functions

- `config.py` — hyperparameters

- `debug.py` — sanity checkers, asserts

---

## 19.7 Sanity Check Checklist

✔ Do your model inputs/outputs have expected shapes?
✔ Are `.requires_grad` flags correctly set?
✔ Is your loss decreasing over time?
✔ Do `.grad` values explode or vanish?
✔ Did you call `.train()` and `.eval()` properly?
✔ Are you detaching everything you log or store?
✔ Are any tensors stuck on CPU while the model is on GPU?

---

## 19.8 Best Practices at a Glance

|Practice	                    |Why it Matters                         |
|-------------------------------|---------------------------------------|
|Always zero gradients	        |Avoid accumulation across batches      |
|Use `.detach()` for logging	    |Avoid unwanted graph retention         |
|Profile early	                |Find slow layers before deployment     |
|Use mixed precision	        |Save memory, speed up training         |
|Assert shapes regularly	    |Prevent silent failures                |
|Avoid silent overfitting	    |Validate early, not just at the end    |


---


## 19.9 Summary

|Tool / Tip	                        |Use Case                               |
|-----------------------------------|---------------------------------------|
|`set_detect_anomaly(True)`	        |Catch bad gradients / in-place ops     |
|`torch.profiler	`               |Pinpoint slow layers                   |
|`.grad.norm()` monitoring	        |Detect exploding/vanishing gradients   |
|`memory_summary()`	                |See where your VRAM is going           |
|Code modularization	            |Keeps training and model logic clean   |

> PyTorch is flexible — but that flexibility means you have to be responsible for sanity. <br>
Trust nothing. Print everything. Profile often.

