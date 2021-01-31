module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Distributed

const X = vcat(ones(Float32, Int((Const.dimB+Const.dimS)/2)), 
              -ones(Float32, Int((Const.dimB+Const.dimS)/2)))

function sampling(ϵ::Float32, lr::Float32)

    # Initialize
    x = shuffle(X)
    xdata = Vector{Vector{Float32}}(undef, Const.iters_num)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0
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
        Func.ANN.backward(x, e)
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    error    = (energy - ϵ)^2 / 2f0

    # Update Parameters
    Func.ANN.update(energy, ϵ, lr)

    # Output
    return error, energyS, energyB, numberB
end

function calculation_energy()

    # Initialize
    x = shuffle(X)
    xdata = Vector{Vector{Float32}}(undef, Const.iters_num)
    energy  = 0f0
    senergy = 0f0
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
        senergy += abs2(eS)
        numberB += sum(x[1:Const.dimB])
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    stdval   = sqrt(senergy - energyS^2)

    return energyS, energyB, numberB, stdval
end

end
