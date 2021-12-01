const AbstractInverse{T, M} = Union{Inverse{T, M}, PseudoInverse{T, M}}
const AdjointInverse{T, M} = Union{Adjoint{T, Inverse{T, M}}, Adjoint{T, PseudoInverse{T, M}}}

import LinearAlgebra: adjoint, transpose
adjoint(Inv::AbstractInverse) = Adjoint(Inv)

import LinearAlgebra: ishermitian, issymmetric
ishermitian(Inv::AbstractInverse) = ishermitian(Inv.parent)
issymmetric(Inv::AbstractInverse) = issymmetric(Inv.parent)

function Base.AbstractMatrix(A::AdjointInverse)
	Inv = A.parent
	B = pseudoinverse(Inv.parent') # automatically returns Inverse if matrix is square
	AbstractMatrix(B)
end
function Base.Matrix(A::AdjointInverse)
	Inv = A.parent
	B = pseudoinverse(Inv.parent') # automatically returns Inverse if matrix is square
	Matrix(B)
end

import LinearAlgebra: diag
diag(Inv::Union{AbstractInverse, AdjointInverse}) = diag(AbstractMatrix(Inv))

#################### Basic multiplication and division #########################
import LinearAlgebra: *, /, \

*(L::AbstractInverse{<:Any, <:Number}, B) = L.parent \ B # returns AbstractInverse
*(L::AbstractInverse, B::Number) = pseudoinverse(L.parent / B) # returns AbstractInverse
*(L::AbstractInverse{<:Any, <:Number}, B::Number) = L.parent \ B # returns AbstractInverse
*(L::AbstractInverse, B::AbstractVector) = L.parent \ B
*(L::AbstractInverse, B::AbstractMatrix) = L.parent \ B
*(L::AbstractInverse, B::Adjoint{Any, <:AbstractMatrix}) = L.parent \ B

*(B, L::AbstractInverse{<:Any, <:Number}) = B / L.parent # returns AbstractInverse
*(B::Number, L::AbstractInverse) = pseudoinverse(B \ L.parent) # returns AbstractInverse
*(B::Number, L::AbstractInverse{<:Any, <:Number}) = B / L.parent # returns AbstractInverse
*(B::Adjoint{<:Any, <:AbstractVector}, L::AbstractInverse) = B / L.parent
*(B::Adjoint{<:Any, <:AbstractMatrix}, L::AbstractInverse) = B / L.parent
*(B::AbstractMatrix, L::AbstractInverse) = B / L.parent

\(L::AbstractInverse{<:Any, <:Number}, B) = L.parent * B
\(L::AbstractInverse, B::AbstractVector) = L.parent * B
\(L::AbstractInverse, B::AbstractMatrix) = L.parent * B
\(L::AbstractInverse, B::Adjoint{<:Any, <:AbstractMatrix}) = L.parent * B

/(B, L::AbstractInverse{<:Any, <:Number}) = B * L.parent
/(B::Adjoint{<:Any, <:AbstractVector}, L::AbstractInverse) = B * L.parent
/(B::Adjoint{<:Any, <:AbstractMatrix}, L::AbstractInverse) = B * L.parent
/(B::AbstractMatrix, L::AbstractInverse) = B * L.parent

# Adjoints of pseudo-inverses
# to avoid ambiguities with LinearAlgebra/src/matmul?
*(B, L::AdjointInverse) = (L'*B')'
*(B::Number, L::AdjointInverse) = (L'*B')' # returns Adjoint of AbstractInverse
*(B::Adjoint{<:Any, <:AbstractVector}, L::AdjointInverse) = (L'*B')'
*(B::Adjoint{<:Any, <:AbstractMatrix}, L::AdjointInverse) = (L'*B')'
*(B::AbstractMatrix, L::AdjointInverse) = (L'*B')'

*(L::AdjointInverse, B) = (B'*L')'
*(L::AdjointInverse, B::Number) = (B'*L')'
*(L::AdjointInverse, B::AbstractVector) = (B'*L')'
*(L::AdjointInverse, B::AbstractMatrix) = (B'*L')'
*(L::AdjointInverse, B::Adjoint{<:Any, <:AbstractMatrix}) = (B'*L')'

\(L::AdjointInverse{<:Any, <:Number}, B) = (B'/L')'
\(L::AdjointInverse, B::AbstractVector) = (B'/L')'
\(L::AdjointInverse, B::AbstractMatrix) = (B'/L')'
\(L::AdjointInverse, B::Adjoint{<:Any, <:AbstractMatrix}) = (B'/L')'

/(B, L::AdjointInverse{<:Any, <:Number}) = (L'\B')'
/(B::Adjoint{<:Any, <:AbstractVector}, L::AdjointInverse) = (L'\B')'
/(B::Adjoint{<:Any, <:AbstractMatrix}, L::AdjointInverse) = (L'\B')'
/(B::AbstractMatrix, L::AdjointInverse) = (L'\B')'

# scaling of AbstractInverse by Number
/(L::AbstractInverse, B::Number) = pseudoinverse(L.parent * B)
/(L::AdjointInverse, B::Number) = (B' \ L')'
\(B::Number, L::AbstractInverse) = pseudoinverse(B * L.parent)
\(B::Number, L::AdjointInverse) = (L' / B')'

# *(L1::Inverse, L2::Inverse) =  Inverse(L1.parent * L2.parent) IDEA: LazyMatrixProduct to avoid O(n^3) multiply
# IDEA: could have check for L.parent ≡ B in multiply, to return identity with O(1) operations
# IDEA: add rdiv!, ldiv! with Number types

##################### in-place multiplication and solving ######################
# TODO: add further tests for mul!, and div! methods (e.g. involving scalar)
import LinearAlgebra: ldiv!, rdiv!, mul!
function ldiv!(Y::AbstractVecOrMat, A::AbstractInverse, B::AbstractVecOrMat)
	mul!(Y, A.parent, B)
end
# left multiplying with inverse
mul!(Y, A::AbstractInverse, B) = ldiv!(Y, A.parent, B) # 5 arg?
function mul!(Y, A::AbstractInverse, B, α::Real)
	mul!(Y, A, B)
	@. Y *= α
end
function mul!(Y, A::AbstractInverse, B, α::Real, β::Real)
	A.parent isa Matrix && throw("in place mul! only works if this only works if A.parent is Factorization OR a special matrix type like Diagonal, Bidiagonal, etc.")
	Z = copy(Y) # IDEA: pre-allocate somewhere?
	mul!(Y, A, B)
	@. Y = α*Y + β*Z
	return Y
end

# right multiplying with inverse
function mul!(Y, A, B::AbstractInverse)
	copy!(Y, A)
	rdiv!(Y, B.parent)
end
function mul!(Y, A, B::AbstractInverse, α::Real)
	mul!(Y, A, B)
	@. Y *= α
end
function mul!(Y, A, B::AbstractInverse, α::Real, β::Real)
	Z = copy(Y) # IDEA: pre-allocate somewhere?
	mul!(Y, A, B)
	@. Y = α*Y + β*Z
	return Y
end

# A \ b in place, overwriting B
lmul!(A::AbstractInverse, B) = ldiv!(A.parent, B) # these are usuall only defined for numbers
rmul!(A, B::AbstractInverse) = rdiv!(A, B.parent)

function ldiv!(A::Inverse, B)
	Y = zero(B)
	mul!(Y, A.parent, B)
end
function rdiv!(A, B::Inverse)
	Y = zero(B)
	mul!(Y, A, B.parent)
end

############################# ternary dot product ##############################
LinearAlgebra.dot(x, A::Inverse, y) = dot(x, A*y)
