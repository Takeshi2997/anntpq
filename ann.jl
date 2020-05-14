module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        global o[i]  = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
        global oe[i] = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
    end
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    func(x::Float32) = x * σ(x)
    layer1 = Dense(Const.layer[1], Const.layer[2], func)
    layer2 = Dense(Const.layer[2], Const.layer[3], func)
    layer3 = Dense(Const.layer[3], Const.layer[4], func)
    layer4 = Dense(Const.layer[4], Const.layer[5])
    f = Chain(layer1, layer2, layer3, layer4)
    p = params(f)
    Network(f, p)
end

network = Network()

function save(filename)

    f = getfield(network, :f)
    @save filename f
end

function load(filename)

    @load filename f
    p = params(f)
    Flux.loadparams!(network.f, p)
end

function forward(x::Vector{Float32})

    return network.f(x)[1] .+ im * network.f(x)[2]
end

realloss(x::Vector{Float32}) = network.f(x)[1]
imagloss(x::Vector{Float32}) = network.f(x)[2]

function backward(x::Vector{Float32}, e::Complex{Float32})

    realgs = gradient(() -> realloss(x), network.p)
    imaggs = gradient(() -> imagloss(x), network.p)
    for i in 1:Const.layers_num
        dw = realgs[network.f[i].W] .+ im * imaggs[network.f[i].W]
        db = realgs[network.f[i].b] .+ im * imaggs[network.f[i].b]
        o[i].W  += conj.(dw)
        o[i].b  += conj.(db)
        oe[i].W += conj.(dw) * e
        oe[i].b += conj.(db) * e
    end
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)

    for i in 1:Const.layers_num
        ΔW = 2.0f0 * (energy - ϵ) * 2.0f0 * 
        real.(oe[i].W .- energy * o[i].W) / Const.iters_num
        Δb = 2.0f0 * (energy - ϵ) * 2.0f0 * 
        real.(oe[i].b .- energy * o[i].b) / Const.iters_num
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
end

end
