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

const oimid = [[zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]),
                zeros(Complex{Float32}, Const.layer[i+1])] for i in 1:Const.layers_num-1]
const oiend =  [zeros(Complex{Float32}, Const.layer[end], Const.layer[end-1]),
                zeros(Complex{Float32}, Const.layer[end], Const.layer[1])]

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)
function initO()
    for i in 1:Const.layers_num-1
        global o[i]  = Layer(oimid[i]...)
        global oe[i] = Layer(oimid[i]...)
    end
    global o[end]  = Layer(oiend...)
    global oe[end] = Layer(oiend...)
end

# Define Network

struct Output{S<:AbstractArray, T<:AbstractArray}
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
    layer = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], tanh)
    end
    layer[end] = Output(Const.layer[end-1], Const.layer[end], Const.layer[1])
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
    for i in 1:Const.layers_num-1
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]) 
        b = Flux.zeros(Const.layer[i+1])
        parameters[i] = [W, b]
    end
    W = Flux.glorot_uniform(Const.layer[end], Const.layer[end-1])
    b = Flux.glorot_uniform(Const.layer[end], Const.layer[1])
    parameters[end] = [W, b]
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method
function forward(x::Vector{Float32})
    out, b = network.f(x)
    B = b * x
    return out[1] + im * out[2] + B[1] + im * B[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw .* e
        o[i].b  += db
        oe[i].b += db .* e
    end
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)
    α = 1f0 / Const.iters_num
    for i in 1:Const.layers_num
        ΔW = α .* 2f0 .* (energy - ϵ) .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* 2f0 .* (energy - ϵ) .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
end

end
