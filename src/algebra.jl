const AbstractInverse{T, M} = Union{Inverse{T, M}, PseudoInverse{T, M}}

import LinearAlgebra: *, /, \
*(L::AbstractInverse, B::AbstractVector) = L.parent \ B
*(L::AbstractInverse, B::AbstractMatrix) = L.parent \ B
*(L::AbstractInverse, B::Factorization) = L.parent \ B

# since left pseudoinverse behaves differently for right multiplication
*(B::AbstractVector, P::PseudoInverse) = B * Matrix(P) # (A = L.parent; (B * inverse(A'A)) * A')
*(B::AbstractMatrix, P::PseudoInverse) = B * Matrix(P) # (A = L.parent; (B * inverse(A'A)) * A')
*(B::Factorization, P::PseudoInverse) = B * Matrix(P)

*(B::AbstractVector, L::Inverse) = B / L.parent
*(B::AbstractMatrix, L::Inverse) = B / L.parent
*(B::Factorization, L::Inverse) = B / L.parent

\(L::AbstractInverse, B::AbstractVector) = L.parent * B
\(L::AbstractInverse, B::AbstractMatrix) = L.parent * B
\(L::AbstractInverse, B::Factorization) = L.parent * B

/(B::AbstractVector, L::AbstractInverse) = B * L.parent
/(B::AbstractMatrix, L::AbstractInverse) = B * L.parent
/(B::Factorization, L::AbstractInverse) = B * L.parent

# Adjoints of pseudo-inverses
*(B::AbstractVector, L::Adjoint{<:Real, <:PseudoInverse}) = (L'*B')'
*(B::AbstractMatrix, L::Adjoint{<:Real, <:PseudoInverse}) = (L'*B')'
*(B::Factorization, L::Adjoint{<:Real, <:PseudoInverse}) = (L'*B')'

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

# IDEA: we could get away with only one additional allocation if x ≠ y
function dot(x::AbstractVector, A::Inverse{<:Any, <:Union{<:Cholesky, <:CholeskyPivoted}}, y::AbstractVector)
	xp = copy(x)
	yp = x ≡ y ? xp : copy(y)
	dot!!(xp, A, yp)
end

# WARNING: overwrites x and y with intermediate results
function dot!!(x::AbstractVector, A::Inverse{<:Any, <:Cholesky}, y::AbstractVector)
	C = A.parent
	inverse_cholesky_dot!!(x, C, y)
end

# WARNING: overwrites x and y with intermediate results
function dot!!(x::AbstractVector, A::Inverse{<:Any, <:CholeskyPivoted}, y::AbstractVector)
	C = A.parent
	permute!(x, C.p)
	x ≡ y || permute!(y, C.p)
	inverse_cholesky_dot!!(x, C, y)
end

# helper function for ternary dot product with inverse cholesky
function inverse_cholesky_dot!!(x, C, y)
	if x ≡ y
		inverse_cholesky_norm!(C, y)
	else
		if length(x) < parallel_threshold
			sequential_inverse_cholesky_dot!!(x, C, y)
		else
			parallel_inverse_cholesky_dot!!(x, C, y)
		end
	end
end

@inline function sequential_inverse_cholesky_dot!!(x, C, y)
	L = C.U' # since getting L causes allocations
	Lx, Ly =  ldiv!(L, x), ldiv!(L, y)
	return dot(Lx, Ly) # sum(abs2, Ly) is ever so slightly faster
end
@inline function parallel_inverse_cholesky_dot!!(x, C, y)
	L = C.U' # since getting L causes allocations
	Lx = @spawn ldiv!(L, x)
	Ly = @spawn ldiv!(L, y)
	Lx, Ly = fetch(Lx), fetch(Ly)
	return dot(Lx, Ly) # sum(abs2, Ly) is ever so slightly faster
end
# helper function for ternary dot product with inverse cholesky
@inline function inverse_cholesky_norm!(C, y)
	L = C.U' # since getting L causes allocations
	Ly = ldiv!(L, y)
	sum(abs2, Ly)
end

# IDEA: could have non-allocating mul! for this
# advantage seems to be less pronounced than for dot
function *(X, A::Inverse{<:Any, <:Union{<:Cholesky, <:CholeskyPivoted}}, Y)
	C = A.parent
	if X ≡ Y'
		Y = copy(Y)
		if C isa CholeskyPivoted
			permute_rows!(Y, C.piv)
		end
		L = C.U'
		LY = ldiv!(L, Y)
		*(LY', LY)
	else
		*(X, C\Y)
	end
end

# diagonal of ternary product,
# IDEA: could dispatch when diag(LazyMatrixProduct{}) is called
function diag_mul(X, A::Inverse{<:Any, <:Union{<:Cholesky, <:CholeskyPivoted}}, Y)
	C = A.parent
	LY = similar(Y)
	if X ≡ Y'
		L = C.U'
		ldiv!(LY, L, Y)
		vec(sum(abs2, LY, dims = 1)) # IDEA: could we overwrite LY?
	else
		diag(*(X, C\Y)) # this could be made a little more efficient
	end
end

########################## used for CholeskyPivoted code #######################
function permute_rows!(X::AbstractMatrix, p)
	for x in eachcol(X) # permuting rows means we have to iterate through columns
		permute!(x, p)
	end
	return X
end
function permute_columns!(X::AbstractMatrix, p)
	for x in eachrow(X) # permuting columns means we have to iterate through rows
		permute!(x, p)
	end
	return X
end
