import LinearAlgebra: dot
# we could get away with only one additional allocation if x ≠ y, but
# this is actually worse for performance and does not allow for parallelism
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
# advantage seems to be less pronounced than for dot, since with matrices,
# BLAS3 is called and already includes parallelism
function *(X, A::Inverse{<:Any, <:Union{<:Cholesky, <:CholeskyPivoted}}, Y)
	inverse_cholesky_mul(X, A.parent, Y)
end

# saving one matrix solve if X == Y'
function inverse_cholesky_mul(X, C, Y)
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

function diag(A::Inverse{<:Any, <:Cholesky})
	inverse_cholesky_diag(A.parent)
end
function diag(A::Inverse{<:Any, <:CholeskyPivoted})
	C = A.parent
	d = inverse_cholesky_diag(C)
	invpermute!(d, C.piv)
end
# computes the diagonal of the inverse of a Cholesky factorization
function inverse_cholesky_diag(C)
	issuccess(C) || throw("cannot invert unsuccessful factorization, C.info = $(C.info)")
	L = C.U'
	invL = inv(L)
	vec(sum(abs2, invL, dims = 1))
end


################## computes diagonal of ternary product ########################
# IDEA: could dispatch when diag(LazyMatrixProduct{}) is called
function diag_mul(X, A::Inverse{<:Any, <:Cholesky}, Y)
	C = A.parent
	inverse_cholesky_diag_mul(X, C, Y)
end
function diag_mul(X, A::Inverse{<:Any, <:CholeskyPivoted}, Y)
	C = A.parent
	CY = copy(Y)
	if X ≡ Y'
		permute_rows!(CY, C.piv)
		d = inverse_cholesky_diag_norm!(C, CY)
	else
		d = inverse_cholesky_diag_mul!(X, C, CY)
	end
end
# leaves type of C unspecified to work with any implementation of a Cholesky factorization
# as long as it has a C.U field and allows for a backslash solve, and issuccess
function inverse_cholesky_diag_mul(X, C, Y)
	issuccess(C) || throw("cannot invert unsuccessful factorization, C.info = $(C.info)")
	CY = copy(Y)
	if X ≡ Y'
		inverse_cholesky_diag_norm!(C, CY)
	else
		inverse_cholesky_diag_mul!(X, C, CY)
	end
end
# WARNING: overwrites Y
function inverse_cholesky_diag_norm!(C, Y)
	issuccess(C) || throw("cannot invert unsuccessful factorization, C.info = $(C.info)")
	L = C.U'
	ldiv!(L, Y)
	vec(sum(abs2, Y, dims = 1))
end
# WARNING: overwrites Y
function inverse_cholesky_diag_mul!(X, C, Y)
	issuccess(C) || throw("cannot invert unsuccessful factorization, C.info = $(C.info)")
	ldiv!(C, Y)
	diag(X*Y)
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
