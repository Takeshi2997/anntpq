module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, Gen, Serialization
using Flux: @functor
using Flux.Optimise: update!

# Initialize Variables

abstract type Parameters end

mutable struct Layer <: Parameters
    W::Array{Float64}
    b::Array{Float64}
end

o = Vector{Parameters}(undef, Const.layers_num)
μ = Vector{Parameters}(undef, Const.layers_num)

function initO()
    for i in 1:Const.layers_num-1
        W = zeros(Float64, Const.layer[i+1], Const.layer[i])
        b = zeros(Float64, Const.layer[i+1])
        global o[i]   = Layer(W, b)
    end
    W = zeros(Float64, Const.layer[end], Const.layer[end-1])
    b = zeros(Float64, Const.layer[end], Const.layer[1])
    global o[end]   = Layer(W, b)
end

function initμ()
    for i in 1:Const.layers_num-1
        W = zeros(Float64, Const.layer[i+1], Const.layer[i])
        b = zeros(Float64, Const.layer[i+1])
        global μ[i] = Layer(W, b)
    end
    W = zeros(Float64, Const.layer[end], Const.layer[end-1])
    b = zeros(Float64, Const.layer[end], Const.layer[1])
    global μ[end] = Layer(W, b)
end

# Define Network

struct Output{S<:AbstractArray,T<:AbstractArray}
  W::S
  b::T
end

function Output(in::Integer, out::Integer, first::Integer;
                initW = Flux.glorot_uniform, initb = Flux.zeros)
    return Output(initW(out, in), initb(out, first))
end

@functor Output

function (a::Output)(x::AbstractArray)
  W, b = a.W, a.b
  W*x, b
end

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

function Network()
    layer = Vector{Any}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        W = zeros(Float64, Const.layer[i+1], Const.layer[i])
        b = zeros(Float64, Const.layer[i+1])
        layer[i] = Dense(W, b, tanh)
    end
    W = zeros(Float64, Const.layer[end], Const.layer[end-1])
    b = zeros(Float64, Const.layer[end], Const.layer[1])
    layer[end] = Output(W, b)
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, p)
end

network = Network()

# Probabilistic model

@gen function model()
    for i in 1:Const.layers_num
        @trace(broadcasted_normal(μ[i].W, ones(Float64, size(μ[i].W))), :network => i => :W)
        @trace(broadcasted_normal(μ[i].b, ones(Float64, size(μ[i].b))), :network => i => :b)
    end
end

# Network Utility

function savedata(filename::String)
    open(io -> serialize(io, μ), filename, "w")
end

function loaddata(filename::String)
    μ_loc = open(deserialize, filename)
    global μ = μ_loc
end

function load(trace::Vector)
    paramset = [param for param in trace]
    p = params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float64})
    out, b = network.f(x)
    B = b * x
    return out[1] + im * out[2] + B[1] + im * B[2]
end

function backward(e::Float64)
    for i in 1:Const.layers_num
        o[i].W += (network.f[i].W .- μ[i].W) * e
        o[i].b += (network.f[i].b .- μ[i].b) * e
    end
end

opt(lr::Float64) = ADAM(lr, (0.9, 0.999))

function update(energy::Float64, ϵ::Float64, lr::Float64)
    α = 1.0 / Const.num_iters
    for i in 1:Const.layers_num
        ΔW = α .* (energy - ϵ) .* o[i].W
        Δb = α .* (energy - ϵ) .* o[i].b
        update!(opt(lr), μ[i].W, ΔW)
        update!(opt(lr), μ[i].b, Δb)
    end
end

end
