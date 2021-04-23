module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CUDA
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize Variables
mutable struct ParamSet{T <: AbstractArray, S <: AbstractArray}
    o::T
    oe::T
    oo::S
end

function ParamSet()
    W = zeros(Complex{Float32}, Const.networkdim)
    S = transpose(W) .* W
    ParamSet(W, W, S)
end

paramset = ParamSet()

function initParamSet()
    W = zeros(Complex{Float32}, Const.networkdim)
    S = transpose(W) .* W
    setfield!(paramset,  :o, W)
    setfield!(paramset, :oe, W)
    setfield!(paramset, :oo, S)
end

# Define Network

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], swish)
    end
    layers[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
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
    dθ = Complex{Float32}[]
    for i in 1:Const.layers_num
        dW = reshape(gs[network.f[i].W], Const.layer[i+1]*Const.layer[i])
        db = gs[network.f[i].b]
        append!(dθ, dW)
        append!(dθ, db)
    end
    setfield!(paramset,  :o, paramset.o  + dθ)
    setfield!(paramset, :oe, paramset.oe + dθ .* e)
    setfield!(paramset, :oo, paramset.oo + transpose(dθ) .* conj.(dθ))
end

opt(lr::Float32) = Descent(lr)

function calc(e::Float32, ϵ::Float32)
    o  = CuArray(paramset.o  ./ Const.iters_num ./ Const.batchsize)
    oe = CuArray(paramset.oe ./ Const.iters_num ./ Const.batchsize)
    oo = CuArray(paramset.oo ./ Const.iters_num ./ Const.batchsize)
    R  = oe - e * o
    S  = oo - transpose(o) .* conj.(o)
    U, Δ, V = svd(S)
    invΔ = Diagonal(1f0 ./ Δ .* (Δ .> 1f-6))
    Δparam = (e - ϵ) .* 2f0 .* real.(V * invΔ * U' * R) |> cpu
    return Δparam
end

function update(Δparamset::Vector, lr::Float32)
    n = 0
    for i in 1:Const.layers_num
        ΔW = reshape(Δparamset[n+1:n+Const.layer[i+1]*Const.layer[i]], Const.layer[i+1], Const.layer[i])
        n += Const.layer[i+1] * Const.layer[i]
        Δb = Δparamset[n+1:n+Const.layer[i+1]]
        n += Const.layer[i+1]
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
end
end
