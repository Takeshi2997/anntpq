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

mutable struct ParamSet{T <: Parameters}
    o::Vector{T}
    oe::Vector{T}
end

function ParamSet()
    p  = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        p[i]  = Params(W, b)
    end
    W = zeros(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end], Const.layer[1])
    p[end]  = Params(W, b)
    ParamSet(p, p)
end

# Define Network

struct Output{S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
end
function Output(in::Integer, in2::Integer, out::Integer;
                initW = Flux.glorot_uniform, initb = Flux.zeros)
  return Output(initW(out, in), initb(out, in2))
end
@functor Output
function (m::Output)(x::AbstractArray)
    W, b = m.W, m.b
    W*x, b
end

mutable struct Network
    f::Flux.Chain
    p::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], swish)
    end
    layers[end] = Output(Const.layer[end-1], Const.layer[1], Const.layer[end])
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
    for i in 1:Const.layers_num-1
        W = Flux.kaiming_normal(Const.layer[i+1], Const.layer[i])
        b = Flux.zeros(Const.layer[i+1])
        parameters[i] = [W, b]
    end
    W = Flux.kaiming_normal(Const.layer[end], Const.layer[end-1])
    b = Flux.kaiming_normal(Const.layer[end], Const.layer[1])
    parameters[end] = [W, b]
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out, b = network.f(x)
    B = b * x
    return B[1] + im * B[2] + Const.λ .* (out[1] + im * out[2])
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        paramset.o[i].W  += dw
        paramset.o[i].b  += db
        paramset.oe[i].W += dw .* e
        paramset.oe[i].b += db .* e
    end
end

function updateparams(energy::Float32, lr::Float32, paramset::ParamSet, Δparamset::Vector)
    for i in 1:Const.layers_num
        oW   = real.(paramset.o[i].W  / Const.iters_num)
        ob   = real.(paramset.o[i].b  / Const.iters_num)
        oeW  = real.(paramset.oe[i].W / Const.iters_num)
        oeb  = real.(paramset.oe[i].b / Const.iters_num)
        ΔW = oeW - energy * oW
        Δb = oeb - energy * ob
        Δparamset[i][1] += ΔW
        Δparamset[i][2] += Δb
    end
end

opt(lr::Float32) = AMSGrad(lr, (0.9, 0.999))

function update(Δparamset::Vector, lr::Float32)
    for i in 1:Const.layers_num
        update!(opt(lr), network.f[i].W, Δparamset[i][1])
        update!(opt(lr), network.f[i].b, Δparamset[i][2])
    end
end
end
