############################ Lazy Inverse Matrix ###############################
```
converts multiplication into a backsolve and vice versa
applications:
- WoodburyIdentity
-  x' A^{-1} y = *(x, Inverse(A), y) can be 2x for cholesky
- Zygote logdet adjoint
- Laplace's approximation
- solves with kronecker products
```
struct Inverse{T, M} <: Factorization{T} # <: AbstractMatrix{T} ? No, since efficient access to elements is not guaranteed
    parent::M
    function Inverse(A)
        if size(A, 1) == size(A, 2)
			T, M = eltype(A), typeof(A)
			new{T, M}(A)
		else
			throw(DimensionMismatch("Input of size $(size(A)) not square"))
		end
    end
end
Base.size(L::Inverse) = size(L.parent)
Base.size(L::Inverse, dim::Integer) = size(L.parent, dim)

function inverse end # smart pseudo-constructor, is only lazy if inverse costs more than O(1)
inverse(Inv::Inverse) = Inv.parent
inverse(x::Union{Number, UniformScaling}) = inv(x)
inverse(A::AbstractMatrix) = all(==(1), size(A)) ? inv(A[1]) : Inverse(A)
inverse(A::Factorization) = Inverse(A)

LinearAlgebra.inv(Inv::Inverse) = Inv.parent
LinearAlgebra.inv(Inv::Inverse{<:Any, <:Factorization}) = AbstractMatrix(Inv.parent) # since inv is expected to return Matrix
Base.copy(A::Inverse) = inverse(copy(A.parent))

function Base.AbstractMatrix(Inv::Inverse)
	A = inv(Inv.parent)
	if A isa Factorization
		AbstractMatrix(A)
	elseif A isa Number
		fill(A, (1, 1))
	else
		A
	end
end
Base.Matrix(Inv::Inverse) = Matrix(AbstractMatrix(Inv))

# factorize the underlying matrix
import LinearAlgebra: factorize, det, logdet, logabsdet, dot
# factorize is used to compute a type which makes it easy to apply the inverse
# therefore, it should be a no-op on Inverse
factorize(Inv::Inverse) = Inv
det(Inv::Inverse) = 1/det(Inv.parent)
logdet(Inv::Inverse) = -logdet(Inv.parent)
function logabsdet(Inv::Inverse)
    l, s = logabsdet(Inv.parent)
    (-l, s)
end
LinearAlgebra.isposdef(A::Inverse) = isposdef(A.parent)

# IDEA: allows for stochastic approximation:
# A Probing Method for Cοmputing the Diagonal of the Matrix Inverse
import LinearAlgebra: diag
diag(Inv::Inverse) = diag(Matrix(Inv))
diag(Inv::Inverse{<:Any, <:Factorization}) = diag(inv(Inv.parent))

################################################################################
import LinearAlgebra: adjoint, transpose, ishermitian, issymmetric
adjoint(Inv::Inverse) = Inverse(adjoint(Inv.parent))
tranpose(Inv::Inverse) = Inverse(tranpose(Inv.parent))
ishermitian(Inv::Inverse) = ishermitian(Inv.parent)
issymmetric(Inv::Inverse) = issymmetric(Inv.parent)
symmetric(Inv::Inverse) = Inverse(Symmetric(Inv.parent))
hermitian(Inv::Inverse) = Inverse(Hermitian(Inv.parent))
