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
l = length(out)
ψall = Vector{Complex{Float32}}(undef, l)
z = 0f0
@simd for i in 1:l
    x = out[i]
    ψ = exp(ANN.forward(x))
    @inbounds ψall[i] = ψ
    global z += abs2(ψ)
end
ψall ./= sqrt(z)

reψ = real.(ψall)
imψ = imag.(ψall)
histogram(reψ, bins=100)
savefig("repsihist.png")
histogram(imψ, bins=100)
savefig("impsihist.png")

