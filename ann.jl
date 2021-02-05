module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, Distributions
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

abstract type Parameters end
mutable struct Layer <: Parameters
    W::Array{Complex{Float32}}
    b::Array{Complex{Float32}}
end

o   = Vector{Parameters}(undef, Const.layers_num)
oe  = Vector{Parameters}(undef, Const.layers_num)
function initO()
    for i in 1:Const.layers_num-1
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]   = Layer(W, b)
        global oe[i]  = Layer(W, b)
    end
    W = zeros(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    global o[end]   = Layer(W, b)
    global oe[end]  = Layer(W, b)
end

# Define Network

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

function Network()
    layer = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], tanh)
    end
    layer[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, p)
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
    out = network.f(x)
    return out[1] + im * out[2]
end

sqnorm(x::Array{Float32}) = sum(abs2, x)
loss(x::Vector{Float32}) = real(forward(x)) + Const.η * sum(sqnorm, Flux.params(network.f))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw .* e
        o[i].b  += db
        oe[i].b += db .* e
    end
    dw = gs[network.f[end].W]
    o[end].W  += dw
    oe[end].W += dw .* e
end

opt(lr::Float32) = Descent(lr)

function update(energy::Float32, ϵ::Float32, lr::Float32)
    α = 1f0 / Const.iters_num
    lr_loc = (lr * (ϵ - energy) - 1f0)
    for i in 1:Const.layers_num
        ΔW = α .* x .* 2f0 .*  real.(oe[i].W .- (ϵ - energy)* o[i].W)
        Δb = α .* x .* 2f0 .*  real.(oe[i].b .- (ϵ - energy)* o[i].b)
        update!(opt(lr_loc), network.f[i].W, ΔW)
        update!(opt(lr_loc), network.f[i].b, Δb)
    end
    ΔW = α .* x .* 2f0 .* real.(oe[end].W .- (ϵ - energy)* o[end].W)
    update!(opt(lr_loc), network.f[end].W, ΔW)
end
end
