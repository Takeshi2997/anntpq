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

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()
    for i in 1:Const.layers_num-1
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Layer(W, b)
        global oe[i] = Layer(W, b)
    end
    W = zeros(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end], Const.layer[1])
    global o[end]  = Layer(W, b)
    global oe[end] = Layer(W, b)
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
    Z, b = network.f(x)
    B = b * x
    return Z[1] + im * Z[2] + B[1] + im * B[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energy::Float32, ϵ::Float32, lr::Float32)
    x = 2f0 * (energy - ϵ)
    α = x / Const.iters_num
    for i in 1:Const.layers_num
        ΔW = α .* 2f0 .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* 2f0 .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW, o[i].W)
        update!(opt(lr), network.f[i].b, Δb, o[i].b)
    end
end

const ϵ = 1f-8

mutable struct QRMSProp
  eta::Float32
  rho::Float32
  acc::IdDict
end

QRMSProp(η = 0.001f0, ρ = 0.9f0) = QRMSProp(η, ρ, IdDict())

function apply!(o::QRMSProp, x, g, O)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * abs2(O)
  @. g *= η / (√acc + ϵ)
end

function update!(opt, x, x̄, x̂)
  x .-= apply!(opt, x, x̄, x̂)
end

function update!(opt, xs::Params, gs, o)
  for x in xs
    gs[x] == nothing && continue
    update!(opt, x, gs[x], o)
  end
end

end
