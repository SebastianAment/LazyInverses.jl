module TestLazyInverses
using LinearAlgebra
using LazyInverses
using Test

@testset "inverse" begin
    n = 3
    element_types = (Float32, Float64, ComplexF32, ComplexF64)
    for elty in element_types
        A = rand(elty, n, n)
        A = A'A
        Inv = inverse(A)
        @test A*Inv ≈ (one(elty)*I)(n)
        @test Inv*A ≈ (one(elty)*I)(n)
        M = inv(A)

        # determinant
        @test det(Inv) ≈ det(M)
        @test logdet(Inv) ≈ logdet(M)
        @test all(logabsdet(Inv) .≈ logabsdet(M))

        # factorize
        @test factorize(Inv) ≡ Inv # no-op
        @test isposdef(Inv)
        @test ishermitian(Inv)

        # inv
        @test inv(Inv) isa AbstractMatrix
        @test inv(Inv) ≈ A
        D = Diagonal(rand(elty, n))
        @test inv(Inverse(D)) isa AbstractMatrix
        @test inv(Inverse(D)) ≈ D
        @test inv(Inverse(Inv)) isa AbstractMatrix
        @test inv(Inverse(Inv)) ≈ inv(A)

        @test AbstractMatrix(Inv) ≈ inv(A)
        @test AbstractMatrix(Inverse(D)) isa Diagonal
        @test Matrix(Inverse(D)) isa Matrix

        x = rand(elty, (1, 1))
        @test inverse(x) isa elty
        @test inverse(x) ≈ inv(x)[1] # smart pseudo-constructor returns scalar on 1x1 matrix

        # scalar
        x = rand(elty)
        @test Inverse(x) * x ≈ 1
        @test inverse(x) ≈ 1/x
        @test AbstractMatrix(Inverse(x)) ≈ fill(1/x, (1, 1))
    end
end

@testset "pseudoinverse" begin
    n, m = 3, 2
    A = randn(n, m)
    a = randn(n)
    b = randn(m)
    LInv = pseudoinverse(A)
    ML = Matrix(LInv)

    @test LInv*A ≈ I(m)
    @test ML*A ≈ I(m)
    @test LInv * a ≈ ML * a
    @test b' * LInv ≈ b' * ML

    A = randn(m, n)
    RInv = pseudoinverse(A, Val(:R))
    MR = Matrix(RInv)
    @test A*RInv ≈ I(m)
    @test A*MR≈ I(m)
    @test RInv * b ≈ MR * b
    @test a' * RInv ≈ a' * MR

    # factorize
    @test factorize(LInv) ≡ LInv
    @test factorize(RInv) ≡ RInv

    A = randn(n, n)
    @test pinverse(A) isa Inverse
end

# testing particularly efficient algebraic operations with inverse(Cholesky)
@testset "cholesky" begin
    n, k = 1024, 128
    x, y = randn(n), randn(n)
    X, Y = randn(k, n), randn(n, k)
    A = randn(n, n)
    A = A'A
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
end

end # TestLazyInverses
