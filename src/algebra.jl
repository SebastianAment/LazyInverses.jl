import LinearAlgebra: adjoint, transpose
adjoint(Inv::Inverse) = inverse(adjoint(Inv.parent))
adjoint(Inv::PseudoInverse) = pseudoinverse(adjoint(Inv.parent))

transpose(Inv::Inverse) = inverse(transpose(Inv.parent))
transpose(Inv::PseudoInverse) = pseudoinverse(transpose(Inv.parent))

import LinearAlgebra: ishermitian, issymmetric
ishermitian(Inv::AbstractInverse) = ishermitian(Inv.parent)
issymmetric(Inv::AbstractInverse) = issymmetric(Inv.parent)

import LinearAlgebra: diag
# IDEA: allows for stochastic approximation:
# A Probing Method for Cοmputing the Diagonal of the Matrix Inverse
diag(Inv::AbstractInverse) = diag(AbstractMatrix(Inv))

#################### Basic multiplication and division #########################
import LinearAlgebra: *, /, \

*(L::AbstractInverse, B::Number) = pseudoinverse(L.parent / B) # returns AbstractInverse
*(L::AbstractInverse{<:Any, <:Number}, B::Number) = L.parent \ B
*(L::AbstractInverse, B::AbstractVector) = L.parent \ B
*(L::AbstractInverse, B::AbstractMatrix) = L.parent \ B
*(L::AbstractInverse, B::Adjoint{Any, <:AbstractMatrix}) = L.parent \ B

*(B::Number, L::AbstractInverse) = pseudoinverse(B \ L.parent) # returns AbstractInverse
*(B::Number, L::AbstractInverse{<:Any, <:Number}) = B / L.parent
*(B::Adjoint{<:Any, <:AbstractVector}, L::AbstractInverse) = B / L.parent
*(B::Adjoint{<:Any, <:AbstractMatrix}, L::AbstractInverse) = B / L.parent
*(B::AbstractMatrix, L::AbstractInverse) = B / L.parent

\(L::AbstractInverse, B::AbstractVector) = L.parent * B
\(L::AbstractInverse, B::AbstractMatrix) = L.parent * B
\(L::AbstractInverse, B::Adjoint{<:Any, <:AbstractMatrix}) = L.parent * B

/(B::Adjoint{<:Any, <:AbstractVector}, L::AbstractInverse) = B * L.parent
/(B::Adjoint{<:Any, <:AbstractMatrix}, L::AbstractInverse) = B * L.parent
/(B::AbstractMatrix, L::AbstractInverse) = B * L.parent

# scaling of AbstractInverse by Number
/(L::AbstractInverse, B::Number) = pseudoinverse(L.parent * B)
\(B::Number, L::AbstractInverse) = pseudoinverse(B * L.parent)

# *(L1::Inverse, L2::Inverse) =  Inverse(L1.parent * L2.parent) IDEA: LazyMatrixProduct to avoid O(n^3) multiply
# IDEA: could have check for L.parent ≡ B in multiply, to return identity with O(1) operations
##################### in-place multiplication and solving ######################
# IDEA: add rdiv!, ldiv! with Number types
# TODO: add further tests for mul!, and div! methods (e.g. involving scalar)
import LinearAlgebra: ldiv!, rdiv!, mul!, rmul!, lmul!
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
	if !(α == one(α) && β == zero(β))
		@. Y = α*Y + β*Z
	end
	return Y
end

# A \ b in place, overwriting B
lmul!(A::AbstractInverse, B) = ldiv!(A.parent, B) # these are usuall only defined for numbers
rmul!(A, B::AbstractInverse) = rdiv!(A, B.parent)

function ldiv!(A::AbstractInverse, B)
	Y = zero(B)
	mul!(Y, A.parent, B)
end
function rdiv!(A, B::AbstractInverse)
	Y = zero(A)
	mul!(Y, A, B.parent)
end

############################# ternary dot product ##############################
LinearAlgebra.dot(x, A::AbstractInverse, y) = dot(x, A*y)
