include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore
using LinearAlgebra, Flux

const state = collect(-Const.dimB+1:2:Const.dimB-1)
const πf0 = Float32(π)

function energy(β)

    ϵ = Const.t * abs.(cos.(πf0 / Const.dimB * state))
    return -sum(ϵ .* tanh.(β * ϵ)) / Const.dimB 
end

function f(t)
    
    ϵ = Const.t * abs.(cos.(πf0 / Const.dimB * state))
    return - t * sum(log.(cosh.(ϵ / t)))
end

function df(t)

    ϵ = Const.t * abs.(cos.(πf0 / Const.dimB * state))
    return sum(-log.(cosh.(ϵ / t)) .+ (ϵ / t .* tanh.(ϵ / t)))
end

function s(u, t)

    return (u - f(t)) / t
end

function ds(u, t)

    return -(u - f(t)) / t^2 - df(t) / t
end

function translate(u)

    outputs = 0.0f0
    t = 5.0f0
    tm = 0.0f0
    tv = 0.0f0
    for n in 1:1
        dt = ds(u, t)
        lr_t = 0.1f0 * sqrt(1.0f0 - 0.999f0^n) / (1.0f0 - 0.9f0^n)
        tm += (1.0f0 - 0.9f0) * (dt - tm)
        tv += (1.0f0 - 0.999f0) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 10.0f0^(-7))
        outputs = s(u, t)
    end

    return 1.0f0 / t
end

function calculate()

    dirname = "./data"
    f = open("energy_data.txt", "w")
    for iϵ in 1:Const.iϵmax

        filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"

        MLcore.Func.ANN.load(filenameparams)

        energyS, energyB, numberB = MLcore.calculation_energy()

        β = translate(energyB - 1.0f0 * Const.dimB)
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(energyS / Const.dimS - 1.0f0/4.0f0))
        write(f, "\t")
        write(f, string(energyB / Const.dimB - 1.0f0))
        write(f, "\t")
        write(f, string(numberB / Const.dimB))
        write(f, "\n")
    end
    close(f)
end

calculate()


