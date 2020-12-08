include("./setup.jl")
include("./ann.jl")
using .Const, .ANN
using LinearAlgebra, Plots, Flux, BSON, StatsBase

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
ψall = Complex{Float32}[]
z = 0f0
for x in out
    ψ = exp(ANN.forward(x))
    push!(ψall, ψ)
    global z += abs2(ψ)
end
ψall ./= sqrt(z)

reψ = fit(Histogram, real.(ψall), nbins=100)
imψ = fit(Histogram, imag.(ψall), nbins=100)
plot(reψ)
savefig("repsihist.png")
plot(imψ)
savefig("impsihist.png")

