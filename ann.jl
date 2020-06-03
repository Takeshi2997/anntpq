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
    output(x::Complex{Float32}) = log(cosh(x))
    layer = Vector{Flux.Dense}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], func)
    end
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    layer[end] = Dense(W, b, output)
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

<<<<<<< HEAD
    W1 = randn(Float32, Const.layer[2], Const.layer[1]) * sqrt(1.0f0 / Const.layer[1])
    W2 = randn(Float32, Const.layer[3], Const.layer[2]) * sqrt(1.0f0 / Const.layer[2])
    W3 = randn(Float32, Const.layer[4], Const.layer[3]) * sqrt(1.0f0 / Const.layer[3])
    b1 = zeros(Float32, Const.layer[2])
    b2 = zeros(Float32, Const.layer[3])
    b3 = zeros(Float32, Const.layer[4])
    p  = params([W1, b1], [W2, b2], [W3, b3])
=======
    parameters = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        W = Flux.glorot_uniform(Const.layer[i+1], Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = Parameters(W, b)
    end
    W = Flux.glorot_uniform(Const.layer[end], Const.layer[end-1]) .* 
    exp.(π*im* rand(Float32, Const.layer[end], Const.layer[end-1]))
    b  = zeros(Complex{Float32}, Const.layer[end])
    parameters[end] = Parameters(W, b)
    p  = params([[parameters[i].W, parameters[i].b] for i in 1:Const.layers_num]...)
>>>>>>> bm_extended
    Flux.loadparams!(network.f, p)
end

function forward(x::Vector{Float32})

    out = network.f(x)
    return sum(out)
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})

<<<<<<< HEAD
    realgs = gradient(() -> realloss(x), network.p)
    imaggs = gradient(() -> imagloss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = realgs[network.f[i].W] .- im * imaggs[network.f[i].W]
        db = realgs[network.f[i].b] .- im * imaggs[network.f[i].b]
=======
    gs = gradient(() -> loss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
>>>>>>> bm_extended
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
<<<<<<< HEAD
    dw = realgs[network.f[end].W] .- im * imaggs[network.f[end].W]
    o[end].W  += dw
    oe[end].W += dw * e
=======
    u = network.f[1:end-1](x)
    v = tanh.(network.f[end].W * u + network.f[end].b)
    dw = transpose(u) .* v
    db = v
    o[end].W  += dw
    oe[end].W += dw * e
    o[end].b  += db
    oe[end].b += db * e
>>>>>>> bm_extended
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energy::Float32, ϵ::Float32, lr::Float32)

<<<<<<< HEAD
    for i in 1:Const.layers_num-1
        ΔW = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
        real.(oe[i].W .- energy * o[i].W) / Const.iters_num
        Δb = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
        real.(oe[i].b .- energy * o[i].b) / Const.iters_num
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
    real.(oe[end].W .- energy * o[end].W) / Const.iters_num
    update!(opt(lr), network.f[end].W, ΔW)
=======
    α = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* real.(oe[i].b .- energy * o[i].b)
        vW = abs2.(o[i].W)
        vb = abs2.(o[i].b)
        update!(opt(lr), network.f[i].W, ΔW, vW)
        update!(opt(lr), network.f[i].b, Δb, vb)
    end
    ΔW = α .* (oe[end].W .- energy * o[end].W)
    Δb = α .* (oe[end].b .- energy * o[end].b)
    vW = conj.(o[end].W) .* o[end].W
    vb = conj.(o[end].b) .* o[end].b
    update!(opt(lr), network.f[end].W, ΔW, vW)
    update!(opt(lr), network.f[end].b, Δb, vb)
end

const ϵ = 1e-8

mutable struct QRMSProp
  eta::Float64
  rho::Float64
  acc::IdDict
end

QRMSProp(η = 0.001, ρ = 0.9) = QRMSProp(η, ρ, IdDict())

function apply!(o::QRMSProp, x, g, Δ)
  η, ρ = o.eta, o.rho
  acc = get!(o.acc, x, zero(x))::typeof(x)
  @. acc = ρ * acc + (1 - ρ) * Δ
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
>>>>>>> bm_extended
end

end
