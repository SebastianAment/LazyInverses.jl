module TestPseudoInverse
using LinearAlgebra
using LazyInverses
using Test

function scalar_tests(A, M, Inv)
    elty = eltype(A)
    x = randn(elty)
    @testset "scalar tests with eltype $elty" begin
        @test Matrix(Inv * x) ≈ M * x
        @test Matrix(x * Inv) ≈ x * M
        @test Matrix(Inv / x) ≈ M / x
        @test Matrix(x \ Inv) ≈ x \ M
        # @test Matrix(Inv \ x) ≈ A * x
        # @test Matrix(x / Inv) ≈ x * A
    end
end

@testset "pseudoinverse" begin
    n, m = 3, 2
    element_types = (Float32, Float64, ComplexF32, ComplexF64)
    for elty in element_types
        A = randn(elty, n, m)
        a = randn(elty, n)
        b = randn(elty, m)
        B = randn(elty, m, n)

        # pseudoinverse
        LInv = pseudoinverse(A)
        @test LInv isa PseudoInverse{<:elty}
        ML = Matrix(LInv)
        @test ML ≈ pinv(A)

        # scalar operations
        scalar_tests(A, ML, LInv)

        # vector operations
        @test LInv * a ≈ ML * a
        @test b' * LInv ≈ b' * ML

        # matrix operations
        @test LInv * A ≈ I(m)
        @test ML * A ≈ I(m)
        @test A * LInv ≈ A * ML
        @test B' * LInv ≈ B' * ML
        @test LInv * B' ≈ ML * B'
        @test B * LInv' ≈ B * ML'

        # factorize
        @test factorize(LInv) ≡ LInv

        # smart constructor
        A = randn(elty, n, n)
        @test pinverse(A) isa Inverse

        x = rand(elty, (1, 1))
        @test pinverse(x) isa elty
        @test pinverse(x) == inv(x[1]) # smart pseudo-constructor returns scalar on 1x1 matrix
        @test pinverse(Inverse(x)) == x[1]

        # scalar
        x = rand(elty)
        PInv = PseudoInverse(x)
        @test PInv * x ≈ 1
        @test x * PInv ≈ 1
        @test pinverse(x) ≈ 1/x
        @test AbstractMatrix(PInv) ≈ fill(1/x, (1, 1))
    end
end

end # module
