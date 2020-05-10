include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, InteractiveUtils

function learning(io::IOStream, ϵ::Float32, lr::Float32, it_num::Integer)

    error   = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    for it in 1:it_num

        # Calculate expected value
        error, energy, energyS, energyB, numberB = MLcore.sampling(ϵ, lr)

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

    return error, energyS, energyB, numberB
end

function main()

    dirname = "./datainit"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    dirnameerror = "./errorinit"
    rm(dirnameerror, force=true, recursive=true)
    mkdir(dirnameerror)

    for iϵ in 1:1
    
        ϵ = 0.4f0 * Const.t * Const.dimB

        filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"

        # Initialize
        error   = 0.0f0
        energy  = 0.0f0
        energyS = 0.0f0
        energyB = 0.0f0
        numberB = 0.0f0
        lr      = 0.001f0
        it_num  = 5000

        # Learning
        filename = dirnameerror * "/error" * lpad(iϵ, 3, "0") * ".txt"
        f = open(filename, "w")
        @time error, energyS, energyB, numberB = learning(f, ϵ, lr, it_num) 
        close(f)

        MLcore.Func.ANN.save(filenameparams)
    end
end

main()

