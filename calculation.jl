include("./setup.jl")
include("./ml_core.jl")
indlude("./legendreTF.jl")
using .Const, .MLcore, .LegendreTF
using LinearAlgebra, Flux

function calculate()

    dirname = "./data"
    f = open("energy_data.txt", "w")
    for iϵ in 1:Const.iϵmax

        filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"

        MLcore.Func.ANN.load(filenameparams)

        energyS, energyB, numberB = MLcore.calculation_energy()

        β = LegendreTF.translate(energyB)
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(energyS / Const.dimS))
        write(f, "\t")
        write(f, string(energyB / Const.dimB))
        write(f, "\t")
        write(f, string(numberB / Const.dimB))
        write(f, "\n")
    end
    close(f)
end

calculate()


