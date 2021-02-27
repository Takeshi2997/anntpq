module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Random, Statistics, Base.Threads

function inv_iterative_method(ϵ::Float32, lr::Float32, dirname::String, it::Integer)
    # Initialize
    error   = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
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
    end

    # Reset ANN Params
    Func.ANN.reset()

    error = ((energyS + energyB) - ϵ)^2
    return error, energyS, energyB, numberB
end

function sampling(ϵ::Float32, lr::Float32)
    # Initialize
    batchenergyS = zeros(Float32, Const.batchsize)
    batchenergyB = zeros(Float32, Const.batchsize)
    batchnumberB = zeros(Float32, Const.batchsize)
    batchresidue = zeros(Float32, Const.batchsize)
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Float32, Const.layer[i+1], Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = [W, W, b, b]
    end
    Δparamset = [param for param in parameters]
    paramsetvec = [Func.ANN.ParamSet() for n in 1:Const.batchsize]

    @threads for n in 1:Const.batchsize
        batchresidue[n],
        batchenergyS[n],
        batchenergyB[n],
        batchnumberB[n] = mcmc(paramsetvec[n], Δparamset, ϵ, lr)
    end
    for i in 1:Const.layers_num
        Δparamset[i][1] ./= Const.batchsize
        Δparamset[i][2] ./= Const.batchsize
        Δparamset[i][3] ./= Const.batchsize
        Δparamset[i][4] ./= Const.batchsize
    end
    residue = mean(batchresidue)
    energyS = mean(batchenergyS)
    energyB = mean(batchenergyB)
    numberB = mean(batchnumberB)
    Func.ANN.update(Δparamset, lr)

    # Output
    return residue, energyS, energyB, numberB
end

const X = vcat(ones(Float32, Int((Const.dimB+Const.dimS)/2)), -ones(Float32, Int((Const.dimB+Const.dimS)/2)))

function mcmc(paramset, Δparamset::Vector, ϵ::Float32, lr::Float32)

    # Initialize
    energy  = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    residue = 0f0
    ϕ = 0f0
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
        e  = eS + eB
        r  = Func.residue(e - ϵ, x)
        energyS += eS
        energyB += eB
        energy  += e
        numberB += sum(x[1:Const.dimB])
        residue += r
        Func.ANN.backward(x, e, paramset)
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    residue  = sqrt(residue  / Const.iters_num)
    numberB /= Const.iters_num

    # Update Parameters
    Func.ANN.updateparams(energy, ϕ, lr, paramset, Δparamset)

    return residue, energyS, energyB, numberB
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
