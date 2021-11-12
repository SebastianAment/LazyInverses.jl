const AbstractInverse{T, M} = Union{Inverse{T, M}, PseudoInverse{T, M}}
const AdjointInverse{T, M} = Union{Adjoint{T, Inverse{T, M}}, Adjoint{T, PseudoInverse{T, M}}}

import LinearAlgebra: adjoint, transpose
adjoint(Inv::AbstractInverse) = Adjoint(Inv)
tranpose(Inv::AbstractInverse) = Transpose(Inv)

import LinearAlgebra: ishermitian, issymmetric
ishermitian(Inv::AbstractInverse) = ishermitian(Inv.parent)
issymmetric(Inv::AbstractInverse) = issymmetric(Inv.parent)

symmetric(A) = Symmetric(A)
symmetric(Inv::Inverse) = Inverse(Symmetric(Inv.parent))

hermitian(A) = Hermitian(A)
hermitian(Inv::Inverse) = Inverse(Hermitian(Inv.parent))

function Base.AbstractMatrix(A::AdjointInverse)
	Inv = A.parent
	B = pseudoinverse(Inv.parent') # automatically returns Inverse if matrix is square
	AbstractMatrix(B)
end
function Base.Matrix(A::AdjointInverse)
	Inv = A.parent
	B = Inv isa Inverse ? inverse(Inv.parent') : pseudoinverse(Inv.parent')
	Matrix(B)
end

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

##################### in-place multiplication and solving ######################
# TODO: tests, mul!, and div! methods involving scalar
import LinearAlgebra: ldiv!, rdiv!, mul!
ldiv!(Y, A::AbstractInverse, B) = mul!(Y, A.parent, B)
mul!(Y, A::AbstractInverse, B) = ldiv!(Y, A.parent, B) # 5 arg?
function mul!(Y, A::AbstractInverse, B, α::Real)
	ldiv!(Y, A.parent, B) # 5 arg?
	@. Y *= α
end
function mul!(Y, A::AbstractInverse, B, α::Real, β::Real)
	Z = copy(Y) # IDEA: pre-allocate somewhere?
	mul!(Y, A, B)
	@. Y = α*Y + β*Z
	return Y
end

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
dot(x, A::Inverse, y) = dot(x, A*y)
