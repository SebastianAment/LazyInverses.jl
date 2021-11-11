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
@btime dot($x, $invC, $y)
@btime dot($x, $C \ $x)
@btime dot($x, $C \ $y)

k = n
X = randn(k, n)
Y = randn(n, k)
println("matrix-matrix-matrix multiplication")
@btime *($X, $invC, $X')
@btime *($X, $invC, $Y)
@btime *($X, $C \ $X')
@btime *($X, $C \ $Y)
