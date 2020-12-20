include("./setup.jl")
include("./ml_core.jl")
include("./ann.jl")
include("./legendreTF.jl")
using .Const, .ANN, .MLcore, .LegendreTF
using LinearAlgebra, Flux, BSON, CuArrays

function test()
    MLcore.Func.ANN.init()
    
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

@time test()


