module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables

abstract type Parameters end
mutable struct Params{S<:AbstractArray, T<:AbstractArray} <: Parameters
    W::S
    b::T
end
mutable struct WaveFunction{S<:Complex} <: Parameters
    ϕ::S
end

o   = Vector{Parameters}(undef, Const.layers_num)
oe  = Vector{Parameters}(undef, Const.layers_num)
ob  = Vector{Parameters}(undef, Const.layers_num)
b   = Parameters

function initO()
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Params(W, b)
        global oe[i] = Params(W, b)
        global ob[i] = Params(W, b)
    end
    global b = WaveFunction(0f0im)
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
    for i in 1:Const.layers_num-1
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
        W = Flux.kaiming_normal(Const.layer[i+1], Const.layer[i]) 
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
        W = Flux.kaiming_normal(Const.layer[i+1], Const.layer[i])
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

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.q)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        o[i].b  += db
        oe[i].W += dw .* e
        oe[i].b += db .* e
        ob[i].W += dw .* forward_b(x) ./ forward(x)
        ob[i].b += db .* forward_b(x) ./ forward(x)
    end
    b.ϕ += forward_b(x) ./ forward(x)
end

opt(lr::Float32) = AdaBelief(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)
    for i in 1:Const.layers_num
        o[i].W  ./= Const.iters_num
        o[i].b  ./= Const.iters_num
        oe[i].W ./= Const.iters_num
        oe[i].b ./= Const.iters_num
        ob[i].W ./= Const.iters_num
        ob[i].b ./= Const.iters_num
    end
    b.ϕ /= Const.iters_num
    for i in 1:Const.layers_num
        ΔW = real.(oe[i].W - (energy - ϵ) * o[i].W) - (real.(ob[i].W) - real.(o[i].W) .* real.(b.ϕ))
        Δb = real.(oe[i].b - (energy - ϵ) * o[i].b) - (real.(ob[i].b) - real.(o[i].b) .* real.(b.ϕ))
        update!(opt(lr), network.g[i].W, ΔW)
        update!(opt(lr), network.g[i].b, Δb)
    end
end

end
