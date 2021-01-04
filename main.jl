using Distributed
@everywhere include("./setup.jl")
@everywhere include("./ml_core.jl")
@everywhere using .Const, .MLcore
@everywhere using Flux

@everywhere function learning(iϵ::Integer, 
                              dirname::String, dirnameerror::String, 
                              lr::Float64, it_num::Integer)
    # Initialize
    error   = 0.0
    energyS = 0.0
    energyB = 0.0
    numberB = 0.0
    MLcore.Func.ANN.loaddata(dirname * "/params_at_000.dat")
    ϵ = - 0.5 * iϵ / Const.iϵmax * Const.t * Const.dimB
    filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".dat"
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

    MLcore.Func.ANN.savedata(filenameparams)
end

function main()

    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)
    dirnameerror = "./error"
    rm(dirnameerror, force=true, recursive=true)
    mkdir(dirnameerror)
    MLcore.Func.ANN.initμ()
    MLcore.Func.ANN.savedata(dirname * "/params_at_000.dat")
    learning(0, dirname, dirnameerror, Const.lr, Const.it_num)
    exit()

    @time pmap(iϵ -> learning(iϵ, dirname, dirnameerror, Const.lr, Const.it_num), 1:Const.iϵmax)
end

main()

