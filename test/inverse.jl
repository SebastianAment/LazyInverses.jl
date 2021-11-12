module TestInverse
using LinearAlgebra
using LazyInverses
using Test

@testset "inverse" begin
    n, k = 3, 2
    element_types = (Float32, Float64, ComplexF32, ComplexF64)
    for elty in element_types
        A = rand(elty, n, n)
        A = A'A
        Inv = inverse(A)
        @test Inv isa Inverse{<:elty}
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
        if elty <: Real
            @test issymmetric(Inv)
        else
            @test !issymmetric(Inv)
        end

        # algebra - multiplication and division with scalars, vectors, and matrices
        # first, test scaling by numbers
        @testset "scalar algebra with eltype $elty" begin
            x = randn(elty)
            # multiplication of Inverse with scalar
            @test Inv * x isa Inverse
            @test Matrix(Inv * x) ≈ M * x
            @test x * Inv isa Inverse
            @test Matrix(x * Inv) ≈ x * M
            # multiplication of adjoint of Inverse with scalar
            @test Inv' * x isa Adjoint{<:Any, <:Inverse}
            @test Matrix(Inv' * x) ≈ M' * x
            @test x * Inv' isa Adjoint{<:Any, <:Inverse}
            @test Matrix(x * Inv') ≈ x * M'

            # division of Inverse by scalar
            @test Inv / x isa Inverse
            @test Matrix(Inv / x) ≈ M / x
            @test x \ Inv isa Inverse
            @test Matrix(x \ Inv) ≈ x \ M
            # division of adjoint of Inverse by scalar
            @test Inv' / x isa Adjoint{<:Any, <:Inverse}
            @test Matrix(Inv' / x) ≈ M' / x
            @test x \ Inv' isa Adjoint{<:Any, <:Inverse}
            @test Matrix(x \ Inv') ≈ x \ M'
        end

        @testset "vector algebra with eltype $elty" begin
            x = randn(elty, n)
            # Inverse, multiplication with vector
            @test Inv * x ≈ A \ x
            @test x' * Inv ≈ x' / A
            # Inverse, multiplication with vector
            @test Inv \ x ≈ A * x
            @test x' / Inv ≈ x' * A

            # tests with adjoint of Inverse
            @test Inv' * x ≈ A' \ x
            @test Inv' \ x ≈ A' * x
            @test x' * Inv' ≈ x' / A'
            @test x' / Inv' ≈ x' * A'
        end

        @testset "matrix algebra with eltype $elty" begin
            X = randn(elty, n, n)
            @test Inv * X ≈ A \ X
            @test Inv \ X ≈ A * X
            @test X * Inv ≈ X / A
            @test X / Inv ≈ X * A
            @test Inv * X' ≈ A \ X'
            @test Inv \ X' ≈ A * X'
            @test X' * Inv ≈ X' / A
            @test X' / Inv ≈ X' * A
        end

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
        @test x * Inverse(x) ≈ 1
        @test inverse(x) ≈ 1/x
        @test AbstractMatrix(Inverse(x)) ≈ fill(1/x, (1, 1))
    end
end

end # module
