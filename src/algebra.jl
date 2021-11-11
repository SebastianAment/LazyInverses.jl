const AbstractInverse{T, M} = Union{Inverse{T, M}, PseudoInverse{T, M}}

import LinearAlgebra: *, /, \
# Basic multiplication and division
*(L::AbstractInverse, B) = L.parent \ B
\(L::AbstractInverse, B) = L.parent * B
*(B, L::AbstractInverse) = B / L.parent
/(B, L::AbstractInverse) = B * L.parent

# Adjoints of pseudo-inverses
*(B, L::Adjoint{<:Any, <:AbstractInverse}) = (L'*B')'
*(L::Adjoint{<:Any, <:AbstractInverse}, B) = (B'*L')'

# to avoid ambiguities with LinearAlgebra/src/matmul?
*(b::AbstractVector, L::Adjoint{<:Any, <:AbstractInverse}) = (L'*b')'
*(L::Adjoint{<:Any, <:AbstractInverse}, b::AbstractVector) = (b'*L')'
*(B::AbstractMatrix, L::Adjoint{<:Any, <:AbstractInverse}) = (L'*B')'
*(L::Adjoint{<:Any, <:AbstractInverse}, B::AbstractMatrix) = (B'*L')'
*(B::Adjoint{<:Any, <:AbstractVector}, L::Adjoint{<:Any, <:AbstractInverse}) = (L'*B')'
*(L::Adjoint{<:Any, <:AbstractInverse}, B::Adjoint{<:Any, <:AbstractVector}) = (B'*L')'
# *(L1::Inverse, L2::Inverse) =  Inverse(L1.parent * L2.parent) IDEA: LazyMatrixProduct

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
