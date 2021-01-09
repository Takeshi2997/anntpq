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

function initparamlist()
    paramlist = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        paramlist[i] = [W, b]
    end
    return paramlist
end
const paramlist = initparamlist()

mutable struct Diff{S <: Parameters}
    o::Vector{S}
    e::Vector{S}
end
function Diff()
    oinit = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        oinit[i] = Layer(paramlist[i]...)
    end
    Diff(oinit, oinit)
end
o = Diff()
function initO()
    for i in 1:Const.layers_num
        o.o[i] = Layer(paramlist[i]...)
        o.e[i] = Layer(paramlist[i]...)
    end
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
    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o.o[i].W  += dw
        o.e[i].W += dw .* e
        o.o[i].b  += db
        o.e[i].b += db .* e
    end
    dw = gs[network.f[end].W]
    o.o[end].W  += dw
    o.e[end].W += dw .* e
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)
    α = 1f0 / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* 2f0 .* (energy - ϵ) .* real.(o.e[i].W .- energy * o.o[i].W)
        Δb = α .* 2f0 .* (energy - ϵ) .* real.(o.e[i].b .- energy * o.o[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = α .* 2f0 .* (energy - ϵ) .* real.(o.e[end].W .- energy * o.o[end].W)
    update!(opt(lr), network.f[end].W, ΔW)
end

end
