module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, CuArrays, CUDAnative
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::CuArray{Complex{Float32}, 2}
    b::CuArray{Complex{Float32}, 1}
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        W = CuArray(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]))
        b = CuArray(zeros(Complex{Float32}, Const.layer[i+1]))
        global o[i]  = Parameters(W, b)
        global oe[i] = Parameters(W, b)
    end
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    layer = Vector{Flux.Dense}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], tanh) |> gpu
    end
    layer[end] = Dense(Const.layer[end-1], Const.layer[end]) |> gpu
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = params(f)
    Network(f, p)
end

network = Network()

function save(filename)

    f = getfield(network, :f)
    @save filename f
end

function load(filename)

    @load filename f
    p = params(f)
    Flux.loadparams!(network.f, p)
end

function init()

    parameters = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = CuArray(Flux.glorot_uniform(Const.layer[i+1], Const.layer[i]))
        b = CuArray(zeros(Float32, Const.layer[i+1]))
        parameters[i] = Parameters(W, b)
    end
    p = params([[parameters[i].W, parameters[i].b] for i in 1:Const.layers_num]...)
    Flux.loadparams!(network.f, p)
end

const d = CuArray([1f0, 1f0im])
const e = CuArray([1f0, 0f0])
const f = CuArray([0f0, 1f0])

function forward(x::CuArray{Float32, 1})

    out = network.f(x)
    return sum(d .* out)
end

realloss(x::CuArray{Float32, 1}) = sum(e .* network.f(x))
imagloss(x::CuArray{Float32, 1}) = sum(f .* network.f(x))

function backward(x::CuArray{Float32, 1}, e::Complex{Float32})

    realgs = gradient(() -> realloss(x), network.p)
    imaggs = gradient(() -> imagloss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = realgs[network.f[i].W] .- im * imaggs[network.f[i].W]
        db = realgs[network.f[i].b] .- im * imaggs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
    dw = realgs[network.f[end].W] .- im * imaggs[network.f[end].W]
    o[end].W  += dw
    oe[end].W += dw * e
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energy::Float32, ϵ::Float32, lr::Float32)

    α = 4.0f0 * (energy - ϵ) / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α * CUDAnative.real.(oe[i].W .- energy .* o[i].W)
        Δb = α * CUDAnative.real.(oe[i].b .- energy .* o[i].b)
        update!(opt(lr), network.f[i].W, ΔW, o[i].W)
        update!(opt(lr), network.f[i].b, Δb, o[i].b)
    end
    ΔW = α .* real.(oe[end].W .- energy * o[end].W)
    update!(opt(lr), network.f[end].W, ΔW, o[end].W)
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
