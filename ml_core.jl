module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, Gen

function sampling(ϵ::Float64, lr::Float64)
    # Initialize
    traces = Vector{Array}(undef, Const.num_iters)
    energyS = 0.0
    energyB = 0.0
    numberB = 0.0
    Func.ANN.initO()

    # Probabilistic Model Sampling
    model_sampling(traces)
    @simd for trace in traces
        eS, eB, nB = quantum_sampling(trace)
        e = eS + eB
        energyS += eS
        energyB += eB
        numberB += nB
        Func.ANN.backward(e)
    end
    energyS /= Const.num_iters
    energyB /= Const.num_iters
    numberB /= Const.num_iters
    error = (energyS + energyB - ϵ)^2 / 2.0
    return error, energyS, energyB, numberB
end

function model_sampling(traces::Vector{Array})
    trace = simulate(Func.ANN.model, ())
    params = Vector{Array}(undef, Const.layers_num)
    for n=1:Const.num_iters
        trace, = mh(trace, select(:network))
        for i in 1:Const.layers_num
            W = trace[:network => i => :W]
            b = trace[:network => i => :b]
            params[i] = [W, b]
        end
        traces[n] = params
    end
end

function quantum_sampling(trace::Vector{Array})
    # Initialize
    x = rand([1.0, -1.0], Const.dimB+Const.dimS)
    xdata = Vector{Vector{Float64}}(undef, Const.iters_num)
    energyS = 0.0
    energyB = 0.0
    numberB = 0.0
    Func.ANN.load(trace)

    # MCMC Start!
    for i in 1:Const.burnintime
        Func.update(x)
    end
    for i in 1:Const.iters_num
        Func.update(x)
        @inbounds xdata[i] = x
    end

    # Calcurate Physical Value
    @simd for x in xdata
        eS = Func.energyS(x)
        eB = Func.energyB(x)
        energyS += eS
        energyB += eB
        numberB += sum(x[1:Const.dimB])
    end
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    return energyS, energyB, numberB
end

function calculation_energy()

    x = rand([1.0, -1.0], Const.dimB+Const.dimS)
    xdata = Vector{Vector{Float64}}(undef, Const.num)
    energy  = 0.0
    energyS = 0.0
    energyB = 0.0
    numberB = 0.0

    for i in 1:Const.burnintime
        Func.update(x)
    end
    for i in 1:Const.num
        Func.update(x)
        @inbounds xdata[i] = x
    end

    @simd for x in xdata
        eS = Func.energyS(x)
        eB = Func.energyB(x)
        e  = eS + eB
        energyS += eS
        energyB += eB
        energy  += e
        numberB += sum(x[1:Const.dimB])
    end
    energy   = real(energy)  / Const.num
    energyS  = real(energyS) / Const.num
    energyB  = real(energyB) / Const.num
    numberB /= Const.num

    return energyS, energyB, numberB
end

end
