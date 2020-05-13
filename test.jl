include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils, Distributed

println(procs())

B = [0  0  0  0
     0  1 -1  0
     0 -1  1  0
     0  0  0  0] ./ 2.0

@time println(eigvals(B))
