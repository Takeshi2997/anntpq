include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils, Distributed

x = rand([1f0, -1f0], Const.dimB+Const.dimS)
α = rand([1f0, -1f0], length(x))

println(size(vcat(x, α)))

