module LazyInverses
using LinearAlgebra
using Base.Threads: @spawn

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const parallel_threshold = 1024 # used to toggle parallel implementation of inverse Cholesky dot product (in algebra.jl)

export inverse, pinverse, pseudoinverse
export Inverse, PseudoInverse

# IDEA: extend Zygote's logdet adjoint with a lazy inverse ...

include("inverse.jl")
include("pseudo_inverse.jl")
include("algebra.jl")
include("cholesky.jl") # contains specializations for inverses of Cholesky factorizations

end # LazyInverses
