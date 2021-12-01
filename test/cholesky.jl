module TestInverseCholeskyDot
using LinearAlgebra
using LazyInverses
using Test

# testing particularly efficient algebraic operations with inverse(Cholesky)
@testset "cholesky" begin
    n, k = 1024, 128
    element_types = (Float32, Float64, ComplexF32, ComplexF64)
    for elty in element_types
        x, y = rand(elty, n), rand(elty, n)
        X, Y = rand(elty, k, n), rand(elty, n, k)
        A = randn(elty, n, n)
        A = A'A + I # for numerical stability, especially for low-precision floats
        # with Cholesky
        C = cholesky(A)
        invC = Inverse(C)
        @test dot(x, invC, x) ≈ dot(x, C\x)
        @test dot(x, invC, y) ≈ dot(x, C\y)
        @test *(X, invC, X') ≈ *(X, C\X')
        @test *(X, invC, Y) ≈ *(X, C\Y)
        @test diag(invC) ≈ diag(inv(A))

        # with pivoted Cholesky
        C = cholesky(A, Val(true))
        invC = Inverse(C)
        @test dot(x, invC, x) ≈ dot(x, C\x)
        @test dot(x, invC, y) ≈ dot(x, C\y)
        @test *(X, invC, X') ≈ *(X, C\X')
        @test *(X, invC, Y) ≈ *(X, C\Y)
        @test diag(invC) ≈ diag(inv(A))
    end
end

end # TestInverseCholeskyDot
