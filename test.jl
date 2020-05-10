include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils

B = [0  0  0  0
     0  1 -1  0
     0 -1  1  0
     0  0  0  0] ./ 2.0

println(eigvals(B))
