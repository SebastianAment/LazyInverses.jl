################################################################################
# this implements the right pseudoinverse
# is defined if A has linearly independent columns
# ⁻¹, ⁺ syntax
struct PseudoInverse{T, M<:AbstractMatOrFac{T}} <: Factorization{T}
    parent::M
end

Base.size(P::PseudoInverse) = size(P.parent')
Base.size(P::PseudoInverse, k::Integer) = size(P.parent', k::Integer)

function Base.Matrix(P::PseudoInverse, side::Union{Val{:L}, Val{:R}} = Val(:L))
    A = Matrix(P.parent)
    if side isa Val{:L}
        inv(A'A) * A'
    else
        A' * inv(A*A')
    end
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
        return inverse(A)
    elseif side isa Val{:L}
        size(A, 1) > size(A, 2) || error("A does not have linearly independent columns")
        return PseudoInverse(A) # left pinv
    else
        size(A, 2) > size(A, 1) || error("A does not have linearly independent rows")
        return PseudoInverse(A')' # right pinv
    end
end

pseudoinverse(A::Union{Number, Diagonal, UniformScaling}) = inv(A)
pseudoinverse(P::PseudoInverse) = P.parent
pseudoinverse(P::Inverse) = P.parent

LinearAlgebra.adjoint(P::PseudoInverse) = Adjoint(P)
LinearAlgebra.transpose(P::PseudoInverse) = Transpose(P)
