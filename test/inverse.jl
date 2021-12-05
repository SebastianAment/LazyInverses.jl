module TestLazyInverses
using LinearAlgebra
using LazyInverses
using Test

@testset "inverse" begin
    n, k = 3, 2
    element_types = (Float32, Float64, ComplexF32, ComplexF64)
    for elty in element_types
        @testset "eltype $elty" begin
            R = randn(elty, n, n)
            matrix_types = ["generic", "pos. def.", "diagonal"]
            matrices = [R'R + I, R + I, Diagonal(randn(elty, n))]  # loop through different matrix types
            for (A, Aty) in zip(matrices, matrix_types)
                @testset "$Aty matrix" begin
                    Inv = inverse(A)
                    @test Inv isa Inverse{<:elty}
                    @test A*Inv ≈ (one(elty)*I)(n)
                    @test Inv*A ≈ (one(elty)*I)(n)

                    M = inv(A)
                    @test Matrix(adjoint(Inv)) ≈ adjoint(M)
                    @test Matrix(transpose(Inv)) ≈ transpose(M)

                    @test Matrix(Inv') ≈ inv(A)'
                    @test AbstractMatrix(Inv') ≈ inv(A)'
                    @test diag(Inv) ≈ diag(inv(A))
                    @test diag(Inv') ≈ diag(inv(A)')

                    isposdef_A = isposdef(A)
                    @test isposdef(Inv) == isposdef_A
                    @test ishermitian(Inv) == ishermitian(A)
                    @test issymmetric(Inv) == issymmetric(A)

                    # determinant
                    @test det(Inv) ≈ det(M)
                    if isposdef_A # log det errors on non-positive definite systems
                        @test logdet(Inv) ≈ logdet(M)
                    end
                    @test all(logabsdet(Inv) .≈ logabsdet(M))

                    # factorize
                    @test factorize(Inv) ≡ Inv # no-op

                    # algebra - multiplication and division with scalars, vectors, and matrices
                    # first, test scaling by numbers
                    @testset "scalar algebra" begin
                        x = randn(elty)
                        # multiplication of Inverse with scalar
                        @test Inv * x isa Inverse
                        @test Matrix(Inv * x) ≈ M * x
                        @test x * Inv isa Inverse
                        @test Matrix(x * Inv) ≈ x * M
                        # multiplication of adjoint of Inverse with scalar
                        @test Matrix(Inv' * x) ≈ M' * x
                        @test Matrix(x * Inv') ≈ x * M'

                        # division of Inverse by scalar
                        @test Inv / x isa Inverse
                        @test Matrix(Inv / x) ≈ M / x
                        @test x \ Inv isa Inverse
                        @test Matrix(x \ Inv) ≈ x \ M
                        # division of adjoint of Inverse by scalar
                        @test Matrix(Inv' / x) ≈ M' / x
                        @test Matrix(x \ Inv') ≈ x \ M'
                    end

                    @testset "vector algebra" begin
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
                        y = randn(elty, n)
                        @test dot(x, Inv, y) ≈ dot(x, A \ y)
                    end

                    @testset "matrix algebra" begin
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

                    @testset "in-place algebra" begin
                        X = randn(elty, n, n)
                        Y = zero(X)
                        InvFact = Inverse(factorize(A))
                        @test mul!(Y, InvFact, X) ≈ InvFact * X
                        # 4 and 5 arg mul!
                        α, β = randn(2)
                        @test mul!(Y, InvFact, X, α) ≈ α * (InvFact * X)
                        Y = randn(elty, n, n)
                        Z = copy(Y)
                        @test mul!(Y, InvFact, X, α, β) ≈ α * (InvFact * X) + β * Z
                        Z = copy(X)
                        @test lmul!(InvFact, Z) ≈ Inv * X
                        Z = copy(X')
                        @test rmul!(Z, InvFact) ≈ X' * Inv

                        Z = copy(Y)
                        @test mul!(Z, X, InvFact) ≈ X * InvFact
                        α, β = randn(2)
                        @test mul!(Y, X, InvFact, α) ≈ α * (X * InvFact)
                        Z = copy(Y)
                        @test mul!(Y, X, InvFact, α, β) ≈ α * (X * InvFact) + β * Z

                        @test ldiv!(Y, Inv, X) ≈ Inv \ X
                        Z = copy(X)
                        @test ldiv!(Inv, Z) ≈ Inv \ X
                        Z = copy(X')
                        @test rdiv!(Z, Inv) ≈ X' / Inv
                    end

                    # inv
                    @test inv(Inv) isa AbstractMatrix
                    @test inv(Inv) ≈ A
                    @test inv(Inverse(Inv)) isa AbstractMatrix
                    @test inv(Inverse(Inv)) ≈ inv(A)
                    @test AbstractMatrix(Inv) ≈ inv(A)
                end # matrices testset
            end # loop over matrices

            # diagonal case
            D = Diagonal(rand(elty, n))
            @test inv(Inverse(D)) isa AbstractMatrix
            @test inv(Inverse(D)) ≈ D
            @test AbstractMatrix(Inverse(D)) isa Diagonal
            @test Matrix(Inverse(D)) isa Matrix

            # 1 x 1 matrix
            x = rand(elty, (1, 1))
            @test inverse(x) isa elty
            @test inverse(x) ≈ inv(x)[1] # smart pseudo-constructor returns scalar on 1x1 matrix

            # scalar
            x = rand(elty)
            @test Inverse(x) * x ≈ 1
            @test x * Inverse(x) ≈ 1
            @test inverse(x) ≈ 1/x
            @test AbstractMatrix(Inverse(x)) ≈ fill(1/x, (1, 1))
        end # elty testset
    end # loop over elty
end

end # module
