include("./setup.jl")
include("./ann.jl")
using .Const, .ANN
using LinearAlgebra, Plots, Flux, BSON, Serialization

N = 24

function repeatperm(n)
    xs = [1f0, -1f0]
    ys::typeof(xs) = []
    out = []

    function perm()
        if length(ys) == n
            push!(out, collect(ys))
        else
            for x = xs
                push!(ys, x)
                perm()
                pop!(ys)
            end
        end
    end

    perm()
    return out
end

dirname = "./data"
f = open("energy_data.txt", "w")
filenameparams = dirname * "/params_at_000.bson"

ANN.load(filenameparams)

out = repeatperm(N)
ψall = []
z = 0f0
for x in out
    ψ = exp(ANN.forward(x))
    push!(ψall, ψ)
    global z += abs2(ψ)
end
ψall ./= sqrt(z)

open(io -> serialize(io, ψall), "psidata.dat", "w")


