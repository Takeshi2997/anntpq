module MLcore
include("../setup.jl")
include("./functions.jl")
using .Const, .Func, Random, Statistics, Base.Threads

function sampling(ϵ::Float32, lr::Float32)
    # Initialize
    batchenergy  = zeros(Float32, Const.batchsize)
    batchenergyS = zeros(Float32, Const.batchsize)
    batchenergyB = zeros(Float32, Const.batchsize)
    batchnumberB = zeros(Float32, Const.batchsize)
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Float32, Const.layer[i+1], Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = [W, W, b, b]
    end
    Δparamset = [param for param in parameters]
    paramsetvec = [Func.ANN.ParamSet() for n in 1:Const.batchsize]

    @threads for n in 1:Const.batchsize
        batchenergy[n], 
        batchenergyS[n],
        batchenergyB[n],
        batchnumberB[n] = mcmc(paramsetvec[n], Δparamset, ϵ, lr)
    end
    energy  = mean(batchenergy)
    energyS = mean(batchenergyS)
    energyB = mean(batchenergyB)
    numberB = mean(batchnumberB)
    for i in 1:Const.layers_num
        Δparamset[i][1] .*= (energy - ϵ) / Const.batchsize
        Δparamset[i][2] .*= (energy - ϵ) / Const.batchsize
        Δparamset[i][3] .*= (energy - ϵ) / Const.batchsize
        Δparamset[i][4] .*= (energy - ϵ) / Const.batchsize
    end
    Func.ANN.update(Δparamset, lr)

    # Output
    return energy, energyS, energyB, numberB
end

const X = vcat(ones(Float32, Int((Const.dimB+Const.dimS)/2)), -ones(Float32, Int((Const.dimB+Const.dimS)/2)))

function mcmc(paramset, Δparamset::Vector, ϵ::Float32, lr::Float32)

    # Initialize
    energyS = 0f0
    energyB = 0f0
    energy  = 0f0
    numberB = 0f0
    x = shuffle(X)
    xdata = Vector{Vector{Float32}}(undef, Const.iters_num)
    
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
        eI = Func.energyI(x)
        e  = eS + eB + eI
        energyS += eS
        energyB += eB
        energy  += e
        numberB += sum(x[1:Const.dimB])
        Func.ANN.backward(x, e - ϵ, paramset)
    end
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    energy   = real(energy)  / Const.iters_num
    numberB /= Const.iters_num

    # Update Parameters
    Func.ANN.updateparams(energy - ϵ, paramset, Δparamset)

    return energy, energyS, energyB, numberB
end

function calculation_energy(num::Integer)

    # Initialize
    batchenergyS = zeros(Float32, Const.batchsize)
    batchenergyB = zeros(Float32, Const.batchsize)
    batchnumberB = zeros(Float32, Const.batchsize)

    @threads for n in 1:Const.batchsize
        batchenergyS[n],
        batchenergyB[n],
        batchnumberB[n] = mcmc_calc(num)
    end
    energyS = mean(batchenergyS)
    energyB = mean(batchenergyB)
    numberB = mean(batchnumberB)

    # Output
    return energyS, energyB, numberB
end

function mcmc_calc(num::Integer)

    # Initialize
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    x = shuffle(X)
    xdata = Vector{Vector{Float32}}(undef, num)
    
    # MCMC Start!
    for i in 1:Const.burnintime
        Func.update(x)
    end
    for i in 1:num
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
        numberB += sum(x[1:Const.dimB])
    end
    energyS  = real(energyS) / num
    energyB  = real(energyB) / num
    numberB /= num

    return energyS, energyB, numberB
end

end
