include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore
using LinearAlgebra, Flux

const state = collect(-Const.dimB+1:2:Const.dimB-1)

function energy(β)

    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return -sum(ϵ .* tanh.(β * ϵ)) / Const.dimB 
end

function f(t)
    
    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return - t * sum(log.(cosh.(ϵ / t)))
end

function df(t)

    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return sum(-log.(cosh.(ϵ / t)) .+ (ϵ / t .* tanh.(ϵ / t)))
end

function s(u, t)

    return (u - f(t)) / t
end

function ds(u, t)

    return -(u - f(t)) / t^2 - df(t) / t
end

function translate(u)

    outputs = 0.0
    t = 5.0
    tm = 0.0
    tv = 0.0
    for n in 1:5000
        dt = ds(u, t)
        lr_t = 0.1 * sqrt(1.0 - 0.999^n) / (1.0 - 0.9^n)
        tm += (1.0 - 0.9) * (dt - tm)
        tv += (1.0 - 0.999) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 1.0 * 10^(-7))
        outputs = s(u, t)
    end

    return 1 / t
end

function exact_energy()

    dirname = "./data"
    f = open("exact_energy.txt", "w")
    for iβ in 1:1000
        β = iβ * 0.01
   
        ϵ = energy(β)

        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                       (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0)) + 0.125))
        write(f, "\t")
        write(f, string(ϵ + 1.0))
        write(f, "\n")
    end
    close(f)
end   

function calculate()

    dirname = "./data"
    f = open("energy_data.txt", "w")
    for iϵ in 1:Const.iϵmax

        filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"

        MLcore.Func.ANN.load(filenameparams)

        energyS, energyB, numberB = MLcore.calculation_energy()

        β = translate(Float64(energyB - Const.dimB * Const.t))
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(energyS / Const.dimS))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                       (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0)) + 
                      0.125))
        write(f, "\t")
        write(f, string(energyB / Const.dimB))
        write(f, "\t")
        write(f, string(numberB / Const.dimB))
        write(f, "\n")
    end
    close(f)
end

calculate()
exact_energy()


