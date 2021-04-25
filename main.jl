include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore
using Flux

function main()

    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)
    dirnameerror = "./error"
    rm(dirnameerror, force=true, recursive=true)
    mkdir(dirnameerror)
    MLcore.Func.ANN.init()
    MLcore.Func.ANN.save(dirname * "/params_at_000.bson")
    learning(0, dirname, dirnameerror, Const.lr)
    map(iϵ -> learning(iϵ, dirname, dirnameerror, Const.lr), 1:Const.iϵmax)
end

function learning(iϵ::Integer, dirname::String, dirnameerror::String, lr::Float32)
    # Initialize
    error   = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    ϵ = -0.4f0 * iϵ / Const.iϵmax * Const.t * Const.dimB
    filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"
    filename = dirnameerror * "/error" * lpad(iϵ, 3, "0") * ".txt"
    MLcore.Func.ANN.load(dirname * "/params_at_000.bson")
    touch(filename)

    # Learning
    for n in 1:Const.it_num
        energy, energyS, energyB, numberB, energyI = MLcore.imaginary(ϵ, lr)
        error = ((energy - ϵ)/(Const.dimB+Const.dimS))^2 / 2f0
        open(filename, "a") do io
            write(io, string(n))
            write(io, "\t")
            write(io, string(error))
            write(io, "\t")
            write(io, string(energyS / Const.dimS))
            write(io, "\t")
            write(io, string(energyB / Const.dimB))
            write(io, "\t")
            write(io, string(numberB / Const.dimB))
            write(io, "\t")
            write(io, string(energyI / Const.dimS))
            write(io, "\n")
        end
    end

    MLcore.Func.ANN.save(filenameparams)
end

main()

