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
    p::Zygote.Params
end

function Network()

    W = randn(Complex{Float32}, Const.layer[2], Const.layer[1])
    b = randn(Complex{Float32}, Const.layer[2])
    a = randn(Complex{Float32}, Const.layer[1])
    p = params(W, b, a)
    Network(W, b, a, p)
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
    setfield!(network, :p, m.p)
end

function init()

    W = randn(Complex{Float32}, Const.layer[2], Const.layer[1]) / sqrt(Float32(Const.layer[1]))
    b = zeros(Complex{Float32}, Const.layer[2])
    a = randn(Complex{Float32}, Const.layer[1])
    p = params(W, b, a)
    setfield!(network, :W, W)
    setfield!(network, :b, b)
    setfield!(network, :a, a)
    setfield!(network, :p, p)
end

function forward(x::Vector{Float32})

    out = network(x)
    return out
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})

    gs = gradient(() -> loss(x), network.p)
    dw = gs[network.W]
    db = gs[network.b]
    da = gs[network.a]
    o.W  += dw
    oe.W += dw * e
    o.b  += db
    oe.b += db * e
    o.a  += da
    oe.a += da * e
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energyS::Float32, energyB::Float32, ϵ::Float32, lr::Float32)

    energy = energyS + energyB
    α = ifelse(lr > 0f0, 4.0f0 * (energy - ϵ), 1f0 * (energyB < 0f0)) / Const.iters_num
    ΔW = α .* (oe.W .- energy * o.W)
    Δb = α .* (oe.b .- energy * o.b)
    update!(opt(lr), network.W, ΔW, o.W)
    update!(opt(lr), network.b, Δb, o.b)
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
