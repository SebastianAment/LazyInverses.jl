# LazyInverses.jl
LazyInverses provides a lazy wrapper for a matrix inverse, akin to Adjoint in Julia Base. See the README for example use cases.

## Installation

Simply type `]` follwed by `add LazyInverses` in the REPL.

## Basic Usage 

The package exports two types `Inverse` and `PseudoInverse`,
and their corresponding "smart" constructors `inverse` and `pinverse`, which 
return the lazy wrappers, unless the input is a number or a 1 x 1 matrix, in which case the inverse is returned directly.
This first example highlights the lazy behavior of `inverse`, and contrasts it to `inv`:
```julia
using LazyInverses
using LinearAlgebra

n = 1024
x = randn(n)
A = randn(n, n)
@time B = inv(A)
  0.058023 seconds (5 allocations: 8.508 MiB)
@time C = inverse(A)
  0.000003 seconds (1 allocation: 16 bytes)
A \ x ≈ C * x
  true
A * x ≈ C \ x
  true
```

## Use Cases

### Energetic Inner Products
In a number of important models, including any that rely on computations with a multivariate Normal distribution,
it is necessary to compute an energetic inner product with an *inverse of a positive semi-definite matrix*.
In order to allow for a particularly efficient implementation of this operation,
LazyInverses.jl extends the ternary dot product like so:
```julia
dot(::AbstractVector, ::Inverse{<:Number, <:Union{<:Cholesky, <:CholeskyPivoted}}, ::AbstractVector)
```
Benchmarking the package's implementation against a naïve implementation,
we observe an up to **7-fold increase in performance**.
```julia
using LazyInverses
using LinearAlgebra
using BenchmarkTools

n = 1024
x = randn(n)
y = randn(n)
A = randn(n, n)
A = A'A
C = cholesky(A)
invC = inverse(C)

println("ternary dot-product multiplication")
@btime dot($x, $invC, $x)
  121.794 μs (3 allocations: 8.16 KiB)
@btime dot($x, $invC, $y)
  172.736 μs (15 allocations: 17.06 KiB)
@btime dot($x, $C \ $x)
  855.008 μs (1 allocation: 8.12 KiB)
@btime dot($x, $C \ $y)
  850.760 μs (1 allocation: 8.12 KiB)
```

Further, we observe a **speed-up of up to a factor of 2** for a ternary matrix multiplication, 
where the middle matrix is an inverse Cholesky factorization.
```julia
k = n
X = randn(k, n)
Y = randn(n, k)
println("ternary multiplication")
@btime *($X, $invC, $X')
  32.755 ms (5 allocations: 16.00 MiB)
@btime *($X, $invC, $Y)
  54.706 ms (4 allocations: 16.00 MiB)
@btime *($X, $C \ $X')
  59.221 ms (4 allocations: 16.00 MiB)
@btime *($X, $C \ $Y)
  56.177 ms (4 allocations: 16.00 MiB)
```
Notably, the implementation takes advantage of threaded parallelism starting at a vector dimension of 1024,
but calls a single-threaded implementation for smaller vectors to minimize constants and maximize performance.

### WoodburyIdentity.jl
Coming soon.

### KroneckerProducts.jl
Coming soon.

### Zygote.jl
Coming soon.

## Notes
The experiments were run on a 2017 MacBook Pro with an i7 dual core and 16 GB of RAM.
