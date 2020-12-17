include("./setup.jl")
include("./ml_core.jl")
include("./ann.jl")
include("./legendreTF.jl")
using .Const, .ANN, .MLcore, .LegendreTF
using LinearAlgebra, Plots, Flux, BSON, StatsPlots

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

function view(N::Integer)
    MLcore.Func.ANN.init()
    # filenameparams = "./data/params_at_000.bson"
    # MLcore.Func.ANN.load(filenameparams)
    
    out = repeatperm(N)
    l = length(out)
    ψall = Vector{Complex{Float32}}(undef, l)
    z = 0f0
    @simd for i in 1:l
        x = out[i]
        ψ = exp(ANN.forward(x))
        @inbounds ψall[i] = ψ
        z += abs2(ψ)
    end
    ψall ./= sqrt(z)
    
    reψ = real.(ψall)
    imψ = imag.(ψall)
    histogram(reψ, title="Histogram of wave function", label="real part", bins=100)
    histogram!(imψ, label="imag part", bins=100)
    savefig("psihist.png")

    f = open("energy_data.txt", "w")
    energyS, energyB, numberB = MLcore.calculation_energy()
    β = LegendreTF.calc_temperature(energyB / Const.dimB)
    # Write energy
    write(f, string(β))
    write(f, "\t")
    write(f, string(energyS / Const.dimS))
    write(f, "\t")
    write(f, string(energyB / Const.dimB))
    write(f, "\t")
    write(f, string(numberB / Const.dimB))
    write(f, "\n")
    close(f)
end

N = 24
@time view(N)




