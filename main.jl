include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore
using Flux

function learning(iϵ::Integer, dirname::String, dirnameerror::String, inv_n::Integer, lr::Float32)
    # Initialize
    error   = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    ϵ = - 0.5f0 * iϵ / Const.iϵmax * Const.t * Const.dimB
    filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"
    filename = dirnameerror * "/error" * lpad(iϵ, 3, "0") * ".txt"
    dirnameonestep = dirnameerror * "/onestep" * lpad(iϵ, 3, "0")
    mkdir(dirnameonestep)
    MLcore.Func.ANN.load_f(dirname * "/params_at_000.bson")
 
    # Learning
    touch(filename)
    for it in 1:inv_n
        MLcore.Func.ANN.load(dirname * "/params_at_000.bson")
        # Calculate expected value
        error, energyS, energyB, numberB = MLcore.inv_iterative_method(ϵ, lr, dirnameonestep, it)
        open(filename, "a") do io
            write(io, string(it))
            write(io, "\t")
            write(io, string(error))
            write(io, "\t")
            write(io, string(energyS / Const.dimS))
            write(io, "\t")
            write(io, string(energyB / Const.dimB))
            write(io, "\t")
            write(io, string(numberB / Const.dimB))
            write(io, "\n")
        end
    end

    MLcore.Func.ANN.save(filenameparams)
end

function main()

    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)
    dirnameerror = "./error"
    rm(dirnameerror, force=true, recursive=true)
    mkdir(dirnameerror)
    MLcore.Func.ANN.init()
    MLcore.Func.ANN.save(dirname * "/params_at_000.bson")
    learning(0, dirname, dirnameerror, 1, -Const.lr)
    map(iϵ -> learning(iϵ, dirname, dirnameerror, Const.inv_n, Const.lr), 1:Const.iϵmax)
end

main()

