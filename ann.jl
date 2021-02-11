module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize
abstract type Parameters end
mutable struct Params{S<:AbstractArray, T<:AbstractArray} <: Parameters
    W::S
    b::T
end
mutable struct WaveFunction{S<:Complex} <: Parameters
    ϕ::S
end

mutable struct ParamSet{T <: Parameters}
    o::Vector{T}
    oe::Vector{T}
    ob::Vector{T}
    b::T
end

function ParamSet()
    o  = Vector{Parameters}(undef, Const.layers_num)
    oe = Vector{Parameters}(undef, Const.layers_num)
    ob = Vector{Parameters}(undef, Const.layers_num)
    b  = Parameters
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Params(W, b)
        global oe[i] = Params(W, b)
        global ob[i] = Params(W, b)
    end
    global b = WaveFunction(0f0im)
    ParamSet(o, oe, ob, b)
end

# Define Network

mutable struct Network
    f::Flux.Chain
    g::Flux.Chain
    p::Zygote.Params
    q::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    layers[1] = Dense(Const.layer[1], Const.layer[2], tanh)
    for i in 2:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], swish)
    end
    layers[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, f, p, p)
end

network = Network()

# Network Utility

function save(filename)
    f = getfield(network, :f)
    @save filename f
end

function load(filename)
    @load filename f
    p = Flux.params(f)
    Flux.loadparams!(network.f, p)
end

function reset()
    g = getfield(network, :g)
    q = Flux.params(g)
    Flux.loadparams!(network.f, q)
end

function init_sub()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]) .+ 0.1f0
        b = Flux.zeros(Const.layer[i+1]) 
        parameters[i] = [W, b]
    end
    paramset = [param for param in parameters]
    q = Flux.params(paramset...)
    Flux.loadparams!(network.g, q)
end

function init()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i])
        b = Flux.zeros(Const.layer[i+1])
        parameters[i] = [W, b]
    end
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out = network.g(x)
    return out[1] + im * out[2]
end

function forward_b(x::Vector{Float32})
    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    gs = gradient(() -> loss(x), network.q)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        paramset.o[i].W  += dw
        paramset.o[i].b  += db
        paramset.oe[i].W += dw .* e
        paramset.oe[i].b += db .* e
        paramset.ob[i].W += dw .* forward_b(x) ./ forward(x)
        paramset.ob[i].b += db .* forward_b(x) ./ forward(x)
    end
    paramset.b.ϕ += forward_b(x) ./ forward(x)
end

opt(lr::Float32) = Descent(lr)

function updateparams(e::Float32, lr::Float32, paramset::ParamSet, Δparamset::Vector)
    for i in 1:Const.layers_num
        paramset.o[i].W  ./= Const.iters_num
        paramset.o[i].b  ./= Const.iters_num
        paramset.oe[i].W ./= Const.iters_num
        paramset.oe[i].b ./= Const.iters_num
        paramset.ob[i].W ./= Const.iters_num
        paramset.ob[i].b ./= Const.iters_num
    end
    paramset.b.ϕ /= Const.iters_num
    for i in 1:Const.layers_num
        Δparamset[i][1] += real.(paramset.oe[i].W - e * paramset.o[i].W) -
        (real.(paramset.ob[i].W) - real.(paramset.o[i].W) .* real.(paramset.b.ϕ))
        Δparamset[i][2] += real.(paramset.oe[i].b - e * paramset.o[i].b) - 
        (real.(paramset.ob[i].b) - real.(paramset.o[i].b) .* real.(paramset.b.ϕ))
    end
end

function update(Δparamset::Vector, lr::Float32)
    for i in 1:Const.layers_num
        ΔW = Δparamset[i][1]
        Δb = Δparamset[i][2]
        update!(opt(lr), network.g[i].W, ΔW)
        update!(opt(lr), network.g[i].b, Δb)
    end
end
end
