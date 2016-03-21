using SparseRelaxation
using Base.Test

# test zero input
@test all(x -> x == 0, gauss_seidel(speye(100), zeros(100), zeros(100)))

# test b = all ones
@test all(x -> x == 1, gauss_seidel(speye(100), zeros(100), ones(100)))

# test convergence
function converges(backwards)
    A = spdiagm(ones(99), 1, 100, 100)
    A = A + transpose(A)
    L = spdiagm(vec(sum(A, 1))) - A
    x = zeros(size(L, 1))
    b = rand(size(L, 1))
    b = b - mean(b)
    x1 = gauss_seidel(L,x,b,backwards)
    norm(b - L*x1) < norm(b - L*x)
end

# check forward convergence
@test converges(true)

#check backwards convergence
@test converges(false)
