using Distributed
@everywhere include("./setup.jl")
@everywhere include("./ml_core.jl")
@everywhere using .Const, .MLcore
@everywhere using Flux

@everywhere function learning(iϵ::Integer, 
                              dirname::String, dirnameerror::String, 
                              lr::Float32, it_num::Integer)
    # Initialize
    error   = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    MLcore.Func.ANN.load(dirname * "/params_at_000.bson")
    ϵ = (-0.3f0 - 0.2f0 * iϵ / Const.iϵmax) * Const.t * Const.dimB
    filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"
    filename = dirnameerror * "/error" * lpad(iϵ, 3, "0") * ".txt"
    dirnameonestep = dirnameerror * "/onestep"
    mkdir(dirnameonestep)

    # Learning
    touch(filename)
    for it in 1:Const.inv_n
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
        # Reset ANN Params
        MLcore.Func.ANN.reset()
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
    map(iϵ -> learning(iϵ, dirname, dirnameerror, Const.lr, Const.it_num), 1:Const.iϵmax)
end

main()

