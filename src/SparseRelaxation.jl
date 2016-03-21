module SparseRelaxation

export gauss_seidel, gauss_seidel!, weighted_jacobi!, weighted_jacobi

# TODO: how to improve locality
# order is different than it should be?
"""
    gauss_seidel(A::SparseMatrixCSC, x::Vector, b::Vector, backwards::Bool = false)

Return a vector that is the result of applying one iteration of Gauss Seidel to
`Ax=b`. This iteration is equivalent to `L(b - Ux)` where `L` is the lower
triangular part of `A` and `U` is the strictly upper triangular part of `A`.
The matrix `A` must be symmetric. If `backwards` is true, then a backwards
sweep is performed.
"""
function gauss_seidel(A :: SparseMatrixCSC, x :: Vector, b :: Vector, backwards :: Bool = false)
    x_new = copy(x)
    gauss_seidel!(A, x_new, b, backwards)
    x_new
end

"""
    gauss_seidel!(A::SparseMatrixCSC, x::Vector, b::Vector, backwards::Bool = false)

In place, more efficient version of `gauss_seidel(A, x, b)`.
"""
function gauss_seidel!{T}( A :: SparseMatrixCSC{T}
                         , x :: Vector{T}
                         , b :: Vector{T}
                         , backwards::Bool = false
                         )
    rows = rowvals(A)
    vals = nonzeros(A)
    m, n = size(A)
    for col in 1:n
        colsum :: T = 0
        diag :: T = 0

        js = nzrange(A, col)
        if backwards
            reverse(js)
        end
        @inbounds for j in js
            row = rows[j]
            val = vals[j]
            if row != col
                colsum += val * x[row]
            else
                diag = val
            end
        end

        @inbounds x[col] = (b[col] - colsum) / diag
    end
end

function weighted_jacobi!(A :: SparseMatrixCSC, x :: Vector, b :: Vector, weight = 2/3)
    Di = spdiagm(map(x -> 1/x, diag(A)))
    R = A - spdiagm(diag(A))
    x = weight * Di * (b - R*x) + (1 - weight) * x
end

function weighted_jacobi(A :: SparseMatrixCSC, x :: Vector, b :: Vector, weight = 2/3)
    x_new = copy(x)
    weighted_jacobi!(A, x_new, b, weight)
    x_new
end

function sor(A :: SparseMatrixCSC, x :: Vector, b :: Vector)
end

end
