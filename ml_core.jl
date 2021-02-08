module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Random

function inv_iterative_method(ϵ::Float32, lr::Float32, dirname::String, it::Integer)
    # Initialize
    error   = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    Func.ANN.init_sub()
    filename = dirname * "/errorstep" * lpad(it, 3, "0") * ".txt"
    touch(filename)

    # Inverse Iterative mathod Start
    for n in 1:Const.it_num
        residue, energyS, energyB, numberB = sampling(ϵ, lr)
        open(filename, "a") do io
            write(io, string(n))
            write(io, "\t")
            write(io, string(residue))
            write(io, "\t")
            write(io, string(energyS / Const.dimS))
            write(io, "\t")
            write(io, string(energyB / Const.dimB))
            write(io, "\t")
            write(io, string(numberB / Const.dimB))
            write(io, "\n")
        end
        if abs(residue) < 0.1f0
            break
        end
    end
    error = (ϵ - (energyS + energyB))^2
    return error, energyS, energyB, numberB
end

const X = vcat(ones(Float32, Int((Const.dimB+Const.dimS)/2)), -ones(Float32, Int((Const.dimB+Const.dimS)/2)))

function sampling(ϵ::Float32, lr::Float32)

    # Initialize
    x = shuffle(X)
    xdata = Vector{Vector{Float32}}(undef, Const.iters_num)
    energy  = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    Func.ANN.initO()

    # MCMC Start!
    for i in 1:Const.burnintime
        Func.update(x)
    end
    for i in 1:Const.iters_num
        Func.update(x)
        @inbounds xdata[i] = x
    end

    # Calcurate Physical Value
    @simd for x in xdata
        eS = Func.energyS(x)
        eB = Func.energyB(x)
        e  = eS + eB
        energyS += eS
        energyB += eB
        energy  += e
        numberB += sum(x[1:Const.dimB])
        Func.ANN.backward(x, e - ϵ)
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num

    # Update Parameters
    Func.ANN.update(energy, ϵ, lr)
    residue = (energy - ϵ) - real(Func.ANN.b.ϕ)

    # Output
    return residue, energyS, energyB, numberB
end

function calculation_energy(num::Integer)

    x = rand([1f0, -1f0], Const.dimB+Const.dimS)
    xdata = Vector{Vector{Float32}}(undef, num)
    energy  = 0f0
    senergy = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0

    for i in 1:Const.burnintime
        Func.update(x)
    end
    for i in 1:num
        Func.update(x)
        @inbounds xdata[i] = x
    end

    @simd for x in xdata
        eS = Func.energyS(x)
        eB = Func.energyB(x)
        e  = eS + eB
        energyS += eS
        energyB += eB
        energy  += e
        senergy += abs2(eS)
        numberB += sum(x[1:Const.dimB])
    end
    energy   = real(energy)  / num
    energy  /= num
    energyS  = real(energyS) / num
    energyB  = real(energyB) / num
    numberB /= num
    variance = sqrt(senergy - energyS^2)

    return energyS, energyB, numberB, variance
end

end
