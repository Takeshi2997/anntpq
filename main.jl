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

    # Learning
    touch(filename)
    for it in 1:it_num
        # Calculate expected value
        open(filename, "a") do io
            error, energyS, energyB, numberB = MLcore.sampling(ϵ, lr)
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

    for iϵ in 1:Const.iϵmax learning(iϵ, dirname, dirnameerror, Const.lr, Const.it_num) end
end

main()

