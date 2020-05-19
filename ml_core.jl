module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Distributed

function sampling(ϵ::Float32, lr::Float32)

    s = rand([1.0f0, -1.0f0], Const.dimS)
    n = rand([1.0f0, -1.0f0], Const.dimB)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    Func.ANN.initO()

    for i in 1:Const.burnintime
        Func.updateS(s, n)
        Func.updateB(s, n)
    end

    for i in 1:Const.iters_num
        Func.updateS(s, n)
        Func.updateB(s, n)

        eS = Func.energyS(s, n)
        eB = Func.energyB(s, n)
        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(n)

        Func.ANN.backward(s, n, e)
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

    s = rand([1.0f0, -1.0f0], Const.dimS)
    n = rand([1.0f0, -1.0f0], Const.dimB)
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    for i in 1:Const.burnintime
        Func.updateS(s, n)
        Func.updateB(s, n)
    end

    for i in 1:Const.iters_num
        Func.updateS(s, n)
        Func.updateB(s, n)

        eS = Func.energyS(s, n)
        eB = Func.energyB(s, n)
        energyS   += eS
        energyB   += eB
        numberB   += sum(n)
    end
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num

    return energyS, energyB, numberB
end

end
