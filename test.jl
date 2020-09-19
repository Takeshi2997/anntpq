include("./setup.jl")
using .Const, LinearAlgebra, InteractiveUtils, Distributed

x = rand([1f0, -1f0], 88)
l = length(x)
y = zeros(Float32, l)
for ix in 1:l
    for iy in 1:l
        ix′    = (ix + iy - 1) % l + 1
        y[ix] += x[ix′] * x[iy]
        println(ix′)
    end
    exit()
end
 
