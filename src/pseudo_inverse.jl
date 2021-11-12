################################################################################
# this implements the right pseudoinverse
# is defined if A has linearly independent columns
# ⁻¹, ⁺ syntax
struct PseudoInverse{T, M} <: Factorization{T}
    parent::M
end
PseudoInverse(A) = PseudoInverse{eltype(A), typeof(A)}(A)

Base.size(P::PseudoInverse) = size(P.parent')
Base.size(P::PseudoInverse, k::Integer) = size(P.parent', k::Integer)
function Base.AbstractMatrix(P::PseudoInverse)
	A = P.parent
    A isa Number ? fill(inv(A), 1, 1) : (A \ I(size(A, 1)))
end
function Base.Matrix(Inv::PseudoInverse)
	M = AbstractMatrix(Inv)
	M isa Matrix ? M : Matrix(M) # since it could be e.g. a Diagonal
end
# Base.Matrix(P::PseudoInverse) = AbstractMatrix(P)
Base.Matrix(A::Adjoint{<:Number, <:PseudoInverse}) = Matrix(A.parent)'
LinearAlgebra.factorize(P::PseudoInverse) = P # same reasoning as for Inverse

# smart constructor
# calls regular inverse if matrix is square
function pseudoinverse end
const pinverse = pseudoinverse
function pseudoinverse(A::AbstractMatOrFac, side::Union{Val{:L}, Val{:R}} = Val(:L))
    if size(A, 1) == size(A, 2)
        inverse(A)
    else
		side isa Val{:L} ? PseudoInverse(A) : PseudoInverse(A')' # right pinv
    end
end
pseudoinverse(A::Union{Number, UniformScaling}) = inv(A)
pseudoinverse(P::PseudoInverse) = P.parent
pseudoinverse(P::Inverse) = inverse(P) # in case P.parent is scalar
