---
hide:
    - toc
---

# Chapter 14: `torch.fft`

> “When you leave time behind and think in frequencies.”

---

## 14.1 What is `torch.fft`?

The `torch.fft` module brings PyTorch into the **frequency domain**, letting you:

- Decompose signals (like audio, images) into sine/cosine waves  
- Denoise data  
- Detect periodic patterns  
- Power audio processing, computer vision, and even quantum simulations  

> It’s the PyTorch equivalent of NumPy’s `np.fft` and is built on highly optimized backend code (MKL, CUFFT).

---

## 14.2 Forward and Inverse FFT

The most basic use case: go to frequency space, and come back.

### ➤ 1D FFT and IFFT

```python
import torch.fft
x = torch.randn(8)
X = torch.fft.fft(x)                 # Frequency domain (complex numbers)
x_reconstructed = torch.fft.ifft(X)  # Back to time domain
```
Result is a complex tensor: real + imaginary parts
> `X.real`, `X.imag`

---

##  14.3 Real FFTs: `rfft()` and `irfft()`
If your input is real-valued (like audio), use real FFTs for speed:
```python
x = torch.randn(8)
X = torch.fft.rfft(x)           # Faster, optimized for real input
x_rec = torch.fft.irfft(X, n=8) # Reconstruct original signal
```
- `rfft()` reduces output size by ~50%
- `irfft()` needs n (original signal length)

---

##  14.4 2D FFTs — Hello, Images
Use `fft2()` and `ifft2()` to process **2D signals** (images, heatmaps, etc.)
```python
img = torch.randn(128, 128)
F_img = torch.fft.fft2(img)
img_back = torch.fft.ifft2(F_img).real  # Often drop imaginary part
```
> You can even apply **frequency masks** (e.g., blur, sharpen, edge-detect) directly in frequency space.

---

##  14.5 Common Functions in torch.fft

|Function	        |Description                |
|-------------------|---------------------------|
|fft()	            |N-point FFT                |
|ifft()	            |Inverse FFT                |
|rfft()	            |Real-input FFT             |
|irfft()	        |Inverse of real FFT        |
|fft2(), ifft2()	|2D FFT and inverse         |
|fftn()	            |N-dimensional FFT          |

---

##  14.6 Use Cases of FFT in Deep Learning

|Application	                |Why FFT?                           |
|-------------------------------|-----------------------------------|
|Audio analysis	                |Detect pitch, noise, rhythm        |
|Image filtering	            |Frequency-based blurs or edges     |
|Signal denoising	            |Filter out high-frequency noise    |
|Physics/finance models	    T   |time-to-frequency domain switching |
|Neural net acceleration	    |Multiply in freq space (FFT Conv)  |

>  Spectral ConvNets? Yep — they multiply weights in the frequency domain.

---

## ⚠️ 14.7 Caveats and Complex Tensor Handling

- Most FFT results are complex tensors
```python
x = torch.fft.fft(signal)
magnitude = x.abs()
phase = x.angle()
```
- ifft() should return back to your original domain — but may differ slightly due to floating-point precision
- rfft() and irfft() require careful dimension tracking


---

##  14.8 Example: Denoising a Signal with FFT
```python
import torch
# Create noisy sine wave
t = torch.linspace(0, 1, 500)
signal = torch.sin(2 * torch.pi * 5 * t) + 0.5 * torch.randn_like(t)

# FFT
F_signal = torch.fft.fft(signal)

# Zero out high frequencies
F_filtered = F_signal.clone()
F_filtered[50:-50] = 0  # Keep low-frequency content

# IFFT to recover signal
denoised = torch.fft.ifft(F_filtered).real
```
> You just built a basic low-pass filter using PyTorch. 😎

---

## ✅ 14.9 Summary

|Task	                |Use This                   |
|-----------------------|---------------------------|
|1D signal analysis	    |fft, rfft, ifft            |
|Image processing	    |fft2, ifft2                |
|Speed + real input	    |rfft, irfft                |
|Custom filters	Modify  |FFT result, then ifft      |
|Neural speedups	    |Spectral convolutions      |

- torch.fft brings NumPy-level spectral power into the PyTorch ecosystem

- All FFT outputs are complex tensors — handle real/imag wisely

- Use this for audio, images, denoising, and modeling periodic signals