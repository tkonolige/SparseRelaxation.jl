# SparseRelaxation.jl

Sparse relaxation methods implemented in Julia. This package borrows heavily from [pyamg](https://github.com/pyamg/pyamg).

## Examples

```julia
A = sprand(10,10,0.1) + spdiagm(ones(10))
x = zeros(10)
b = rand(10)

gauss_seidel(A,x,b) # Smooths on Ax=b using forward Gauss-Seidel smoothing

weighted_jacobi(A,x,b) # Use weighted Jacobi smoothing instead

gauss_seidel!(A,x,b) # in place version
```
