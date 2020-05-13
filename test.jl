include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils, Distributed


B = [1  0  0  0
     0 -1  2  0
     0  2 -1  0
     0  0  0  1]

@time println(eigvals(B))
