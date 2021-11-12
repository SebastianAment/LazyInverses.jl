using LazyInverses
using LinearAlgebra
using BenchmarkTools

n = 1024
x = randn(n)
A = randn(n, n)
@time B = inv(A)
@time C = inverse(A)
println(A \ x ≈ C * x)
println(A * x ≈ C \ x)
