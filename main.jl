include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, InteractiveUtils
using Flux

function learning(filename::String, ϵ::Float32, lr::Float32, it_num::Integer)

    error   = 0f0
    energyS = 0f0
    energyB = 0f0
    numberB = 0f0

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

    return error, energyS, energyB, numberB
end

function main()

    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    dirnameerror = "./error"
    rm(dirnameerror, force=true, recursive=true)
    mkdir(dirnameerror)

    MLcore.Func.ANN.init()

    g = open("error.txt", "w")
    for iϵ in 1:Const.iϵmax

        ϵ = - 0.5f0 * iϵ / Const.iϵmax * Const.t * Const.dimB
        filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"

        # Initialize
        error   = 0f0
        energy  = 0f0
        energyS = 0f0
        energyB = 0f0
        numberB = 0f0
        lr      = Const.lr
        it_num  = Const.it_num

        # Learning
        filename = dirnameerror * "/error" * lpad(iϵ, 3, "0") * ".txt"
        @time error, energyS, energyB, numberB = learning(filename, ϵ, lr, it_num) 

        # Write error
        write(g, string(iϵ))
        write(g, "\t")
        write(g, string(error))
        write(g, "\t")
        write(g, string(energyS / Const.dimS))
        write(g, "\t")
        write(g, string(energyB / Const.dimB))
        write(g, "\t")
        write(g, string(numberB / Const.dimB))
        write(g, "\n")

        MLcore.Func.ANN.save(filenameparams)
    end
    close(g)
end

main()

