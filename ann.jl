module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        global o[i]  = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
        global oe[i] = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
    end
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    func(x::Float32) = tanh(x)
    layer = Vector{Flux.Dense}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], func)
    end
    layer[end] = Dense(Const.layer[end], Const.layer[end-1])
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
        W = randn(Float32, Const.layer[i+1], Const.layer[i]) / sqrt(Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = Parameters(W, b)
    end
    p = params([[parameters[i].W, parameters[i].b] for i in 1:Const.layers_num]...)
    Flux.loadparams!(network.f, p)
end

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
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
    dw = gs[network.f[end].W]
    o[end].W  += dw
    oe[end].W += dw * e
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energy::Float32, ϵ::Float32, lr::Float32)

    α = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW, o[i].W)
        update!(opt(lr), network.f[i].b, Δb, o[i].b)
    end
    ΔW = α .* (oe[end].W .- energy * o[end].W)
    update!(opt(lr), network.f[end].W, ΔW, o[end].W)
end

const ϵ = 1e-8

mutable struct QRMSProp
  eta::Float64
  rho::Float64
  acc::IdDict
end

QRMSProp(η = 0.001, ρ = 0.9) = QRMSProp(η, ρ, IdDict())

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
