include("./setup.jl")
include("./ml_core.jl")
include("./ann.jl")
using .Const, .ANN, .MLcore
using LinearAlgebra, Plots, Flux, BSON, StatsPlots

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

MLcore.Func.ANN.init()

out = repeatperm(N)
ψall = Complex{Float32}[]
z = 0f0
for x in out
    ψ = exp(ANN.forward(x))
    push!(ψall, ψ)
    global z += abs2(ψ)
end
ψall ./= sqrt(z)

reψ = real.(ψall)
imψ = imag.(ψall)
histogram(reψ, bins=100)
savefig("repsihist.png")
histogram(imψ, bins=100)
savefig("impsihist.png")

