# Chapter 11: `torch.linalg`

> “With great matrices comes great responsibility.”

---

## 11.1 What is `torch.linalg`?

`torch.linalg` is PyTorch’s modern **linear algebra module**, introduced to offer more **stable**, **consistent**, and **NumPy-compatible** operations compared to older functions like `torch.svd()` or `torch.eig()`.

It handles:

- Matrix decompositions  
- Solvers  
- Norms  
- Eigenvalues  
- Inverses  
- Determinants  

> 💡 Bonus: most functions support **batched operations**, **autograd**, and **GPU acceleration**.

---

## 11.2 Matrix Inversion and Solving Linear Systems

### ➤ Invert a matrix

```python
A = torch.randn(3, 3)
A_inv = torch.linalg.inv(A)
```
> ⚠️ Inverting is expensive. If you're solving `Ax = b`, use a solver instead.

### ➤ Solve linear system `Ax = b`
```python
A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
b = torch.tensor([9.0, 8.0])
x = torch.linalg.solve(A, b)
```
> ✅ Preferred over computing inv(A) @ b — more numerically stable.

---

## 11.3 Matrix Determinant and Rank

### ➤ Determinant
```python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
torch.linalg.det(A)
```

### ➤ Matrix Rank
```python
torch.linalg.matrix_rank(A)
```
> Useful for checking if a matrix is invertible or has full column rank.

---

## 11.4 Norms and Condition Numbers

### ➤ Norms
```python
x = torch.tensor([3.0, 4.0])
torch.linalg.norm(x)             # L2 norm

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
torch.linalg.norm(A, ord='fro')  # Frobenius norm
```
> Other `ord` options: `'nuc'` (nuclear), `1`, `2`, `inf`, `-inf`

### ➤ Condition Number
```python
torch.linalg.cond(A)
```
> High condition numbers indicate numerical instability — useful for diagnostics.

---

## 11.5 Eigenvalues and Eigenvectors

### ➤ Eigendecomposition
```python
A = torch.tensor([[5.0, 4.0], [1.0, 2.0]])
eigenvalues, eigenvectors = torch.linalg.eig(A)
```
> ⚠️ eigenvalues may be complex64 or complex128.

### ➤ For symmetric/Hermitian matrices:
```python
eigenvalues = torch.linalg.eigvalsh(A)  # Faster and more stable
```

---

##  11.6 SVD (Singular Value Decomposition)

Useful in:
- PCA
- Low-rank approximation
- Image compression
```python
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
A_reconstructed = U @ torch.diag(S) @ Vh
```

---

## 11.7 QR and LU Decomposition

### ➤ QR decomposition
```python
Q, R = torch.linalg.qr(A)
```

### ➤ LU decomposition
```python
LU, pivots = torch.linalg.lu_factor(A)
```

---

## 11.8 Relationship with NumPy
PyTorch’s `torch.linalg` mirrors `numpy.linalg` in:  
- Function names  
- Shape conventions  
- Numerical semantics

But PyTorch:  
- ✅ Supports autograd  
- ✅ Runs on GPU  
- ✅ Handles batch operations  

> You can almost always port NumPy linear algebra code directly to PyTorch with minimal edits.

---

## ✅ 11.9 Summary
|Operation	            |Function                      |
|-----------------------|------------------------------|
|Matrix inverse	        |torch.linalg.inv()            |
|Solve Ax = b	        |torch.linalg.solve()          |
|Determinant	        |torch.linalg.det()            |
|Eigenvalues/vectors	|torch.linalg.eig()            |
|SVD	                |torch.linalg.svd()            |
|Norm	                |torch.linalg.norm()           |
|R / LU	                |torch.linalg.qr(), lu()       |

- Use torch.linalg for numerically stable, batched, and autograd-compatible operations.

- Avoid inv() when you can use solve().

- Most functions support GPU — just move your tensors to cuda.
