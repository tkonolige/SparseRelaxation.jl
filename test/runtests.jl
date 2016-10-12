using SparseRelaxation
using Base.Test

function conv_test(smoother)
    # test zero input
    @test all(x -> x == 0, smoother(speye(100), zeros(100), zeros(100)))

    # test convergence
    function converges()
        A = spdiagm(ones(99), 1, 100, 100)
        A = A + transpose(A)
        L = spdiagm(vec(sum(A, 1))) - A
        x = zeros(size(L, 1))
        b = rand(size(L, 1))
        b = b - mean(b)
        x1 = smoother(L,x,b)
        norm(b - L*x1) < norm(b - L*x)
    end

    # check convergence
    @test converges()
end

@testset "gauss_seidel forwards" begin
    conv_test((A,x,b)->gauss_seidel(A,x,b,false))
end
@testset "gauss_seidel backwards" begin
    conv_test((A,x,b)->gauss_seidel(A,x,b,true))
end
@testset "weighted_jacobi" begin
    conv_test(weighted_jacobi)
end
