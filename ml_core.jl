module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Distributed

@everywhere function sampling(ϵ::Float32, lr::Float32)

    x = rand([1.0f0, -1.0f0], Const.dimB+Const.dimS)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    Func.ANN.initO()

    for i in 1:Const.burnintime

        Func.update(x)
    end

    for i in 1:Const.iters_num
        Func.update(x)

        eS = Func.energyS(x)
        eB = Func.energyB(x)
        eI = Func.energyI(x)
        e  = eS + eB + eI
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(x[1:Const.dimB])

        Func.ANN.backward(x, e)
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    error    = (energy - ϵ)^2

    Func.ANN.update(energy, ϵ, lr)

    return error, energy, energyS, energyB, numberB
end

function calculation_energy()

    x = ones(Float32, Const.dimB+Const.dimS)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    for i in 1:Const.burnintime

        Func.update(x)
    end

    for i in 1:Const.num
        Func.update(x)

        eS = Func.energyS_shift(x)
        eB = Func.energyB_shift(x)
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
