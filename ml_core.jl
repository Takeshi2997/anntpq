module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Random

function sampling(ϵ::Float32, lr::Float32)

    x = rand([1f0, -1f0], Const.dimB+Const.dimS)
    l = length(x)
    α  = rand([1f0, -1f0], l)
    α′ = rand([1f0, -1f0], l)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    Func.ANN.initO()

    rng = MersenneTwister(1234)
    randarray = rand(rng, Float32, (Const.burnintime+Const.iters_num, 3*l))

    for i in 1:Const.burnintime
        randvec = @view randarray[i, :]
        Func.update(x, α, α′, randvec)
    end

    for i in 1:Const.iters_num
        randvec = @view randarray[Const.burnintime+i, :]
        Func.update(x, α, α′, randvec)

        eS = Func.energyS(x, α, α′)
        eB = Func.energyB(x, α, α′)
        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(x[1:Const.dimB])

        Func.ANN.backward(x, α, e)
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    error    = (energy - ϵ)^2

    Func.ANN.update(energy, ϵ, lr)

    return error, energyS, energyB, numberB
end

function calculation_energy()

    x = rand([1f0, -1f0], Const.dimB+Const.dimS)
    l = length(x)
    α  = rand([1f0, -1f0], l)
    α′ = rand([1f0, -1f0], l)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    rng = MersenneTwister(1234)
    randarray = rand(rng, Float32, (Const.burnintime+Const.iters_num, 2*l))

    for i in 1:Const.burnintime
        randvec = @view randarray[i, :]
        Func.update(x, α, α′, randvec)
    end

    for i in 1:Const.num
        randvec = @view randarrau[Const.burnintime+i, :]
        Func.update(x, α, α′, randvec)
 
        eS = Func.energyS(x, α, α′)
        eB = Func.energyB(x, α, α′)
        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(x[1:Const.dimB])
    end
    energy   = real(energy)  / Const.num
    energyS  = real(energyS) / Const.num
    energyB  = real(energyB) / Const.num
    numberB /= Const.num

    return energyS, energyB, numberB
end

end
