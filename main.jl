include("./setup.jl")
include("./ml_core.jl")
using Distributed
@everywhere using .Const, .MLcore
@everywhere using Flux

@everywhere function learning(iϵ::Integer, direname::String, lr::Float32, it_num::Integer)

    # Initialize
    error   = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0
    MLcore.Func.ANN.init()

    ϵ = - 0.5f0 * iϵ / Const.iϵmax * Const.t * Const.dimB
    filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"
    filename = dirnameerror * "/error" * lpad(iϵ, 3, "0") * ".txt"
 
    # Learning
    io = open(filename, "w")
    for it in 1:it_num

        # Calculate expected value
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
    close(io)

    MLcore.Func.ANN.save(filenameparams)
end

function main()

    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    dirnameerror = "./error"
    rm(dirnameerror, force=true, recursive=true)
    mkdir(dirnameerror)

    lr      = Const.lr
    it_num  = Const.it_num

    @time pmap(iϵ -> learning(iϵ, dirname, lr, it_num), 1:Const.iϵmax)
end

main()

