module TestLazyInverses
using LinearAlgebra
using LazyInverses
using Test

@testset "inverse" begin
    n = 3
    A = randn(n, n)
    A = A'A
    Inv = inverse(A)
    @test A*Inv ≈ I(n)

    # determinant
    @test det(Inv) ≈ 1/det(A)
    @test logdet(Inv) ≈ -logdet(A)
    @test all(logabsdet(Inv) .≈ (-1, 1) .* logabsdet(A))

    # factorize
    @test factorize(Inv) ≡ Inv # no-op
    @test isposdef(Inv)

    # inv
    @test inv(Inv) isa AbstractMatrix
    @test inv(Inv) ≈ A
    D = Diagonal(randn(n))
    @test inv(Inverse(D)) isa AbstractMatrix
    @test inv(Inverse(D)) ≈ D
    @test inv(Inverse(Inv)) isa AbstractMatrix
    @test inv(Inverse(Inv)) ≈ inv(A)

    @test AbstractMatrix(Inv) ≈ inv(A)
    @test AbstractMatrix(Inverse(D)) isa Diagonal
    @test Matrix(Inverse(D)) isa Matrix

    x = randn((1, 1))
    @test inverse(x) isa Real
    @test inverse(x) ≈ inv(x)[1]
end

@testset "pseudoinverse" begin
    A = randn(3, 2)
    LInv = pseudoinverse(A)
    @test LInv*A ≈ I(2)
    @test Matrix(LInv)*A ≈ I(2)

    A = randn(2, 3)
    RInv = pseudoinverse(A, Val(:R))
    @test A*RInv ≈ I(2)
    @test A*Matrix(RInv) ≈ I(2)

    # factorize
    @test factorize(LInv) ≡ LInv
end

# testing particularly efficient algebraic operations with inverse(Cholesky)
@testset "cholesky" begin
    n, k = 1024, 128
    x, y = randn(n), randn(n)
    X, Y = randn(k, n), randn(n, k)

    A = randn(n, n)
    A = A'A
    C = cholesky(A)
    invC = inverse(C)

    @test dot(x, invC, x) ≈ dot(x, C\x)
    @test dot(x, invC, y) ≈ dot(x, C\y)


    @test *(X, invC, X') ≈ *(X, C\X')
    @test *(X, invC, Y) ≈ *(X, C\Y)

    # with pivoted cholesky
    C = cholesky(A, Val(true))
    invC = Inverse(C)
    @test dot(x, invC, x) ≈ dot(x, C\x)
    @test dot(x, invC, y) ≈ dot(x, C\y)
    @test *(X, invC, X') ≈ *(X, C\X')
    @test *(X, invC, Y) ≈ *(X, C\Y)
end

end # TestLazyInverses
