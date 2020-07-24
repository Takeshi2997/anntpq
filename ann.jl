module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, Serialization

mutable struct Parameters

    W::Array
    b::Array
    a::Array
end

o  = Parameters
oe = Parameters

function initO()

    W = zeros(Complex{Float32}, Const.layer[2], Const.layer[1])
    b = zeros(Complex{Float32}, Const.layer[2])
    a = zeros(Complex{Float32}, Const.layer[1])
    global o  = Parameters(W, b, a)
    global oe = Parameters(W, b, a)
end

mutable struct Network

    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
    a::Array{Complex{Float32}, 1}
end

function Network()

    W = randn(Complex{Float32}, Const.layer[2], Const.layer[1])
    b = randn(Complex{Float32}, Const.layer[2])
    a = randn(Complex{Float32}, Const.layer[1])
    Network(W, b, a)
end

(m::Network)(x::Vector{Float32}) = sum(log.(cosh.(m.W * x .+ m.b))) + transpose(m.a) * x

network = Network()

function save(filename)

    open(io -> serialize(io, network), filename, "w")
end

function load(filename)

    m = open(deserialize, filename)
    setfield!(network, :W, m.W)
    setfield!(network, :b, m.b)
    setfield!(network, :a, m.a)
end

function init()

    W = randn(Complex{Float32}, Const.layer[2], Const.layer[1]) / sqrt(Float32(Const.layer[1]))
    b = zeros(Complex{Float32}, Const.layer[2])
    a = randn(Complex{Float32}, Const.layer[1])
    setfield!(network, :W, W)
    setfield!(network, :b, b)
    setfield!(network, :a, a)
end

function forward(x::Vector{Float32})

    out = network(x)
    return out
end

function backward(x::Vector{Float32}, e::Complex{Float32})

    v  = tanh.(network.W * x .+ network.b)
    dW = transpose(x) .* v
    db = v
    da = x
    o.W  += dW
    oe.W += dW * e
    o.b  += db
    oe.b += db * e
    o.a  += da
    oe.a += da * e
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energy::Float32, ϵ::Float32, lr::Float32)

    α = 4.0f0 * (energy - ϵ) / Const.iters_num
    ΔW = α .* (oe.W .- energy * o.W)
    Δb = α .* (oe.b .- energy * o.b)
    Δa = α .* (oe.a .- energy * o.a)
    update!(opt(lr), network.W, ΔW, o.W)
    update!(opt(lr), network.b, Δb, o.b)
    update!(opt(lr), network.a, Δa, o.a)
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
