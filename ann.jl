module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array
    b::Array
end

o1  = Vector{Parameters}(undef, Const.layers1_num)
o1e = Vector{Parameters}(undef, Const.layers1_num)
o2  = Vector{Parameters}(undef, Const.layers2_num)
o2e = Vector{Parameters}(undef, Const.layers2_num)

function initO()

    for i in 1:Const.layers1_num
        W = zeros(Complex{Float32}, Const.layer1[i+1], Const.layer1[i])
        b = zeros(Complex{Float32}, Const.layer1[i+1])
        global o1[i]  = Parameters(W, b)
        global o1e[i] = Parameters(W, b)
    end
    for i in 1:Const.layers2_num
        W = zeros(Complex{Float32}, Const.layer2[i+1], Const.layer2[i])
        b = zeros(Complex{Float32}, Const.layer2[i+1])
        global o2[i]  = Parameters(W, b)
        global o2e[i] = Parameters(W, b)
    end
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
    g::Flux.Chain
    q::Zygote.Params
end

function Network()

    func(x::Float32) = tanh(x)

    # layers1 Def
    layer1 = Vector{Any}(undef, Const.layers1_num)
    for i in 1:Const.layers1_num-1
        layer1[i] = Dense(Const.layer1[i], Const.layer1[i+1], func)
    end
    W1 = randn(Complex{Float32}, Const.layer1[end], Const.layer1[end-1])
    b1 = zeros(Complex{Float32}, Const.layer1[end])
    layer1[end] = Dense(W1, b1)

    # layers2 Def
    layer2 = Vector{Any}(undef, Const.layers2_num)
    for i in 1:Const.layers2_num-1
        layer2[i] = Dense(Const.layer2[i], Const.layer2[i+1], func)
    end
    W2 = randn(Complex{Float32}, Const.layer2[end], Const.layer2[end-1])
    b2 = zeros(Complex{Float32}, Const.layer2[end])
    layer2[end] = Dense(W2, b2)

    # Network Def
    f = Chain([layer1[i] for i in 1:Const.layers1_num]...)
    p = params(f)
    g = Chain([layer2[i] for i in 1:Const.layers2_num]...)
    q = params(g)
    Network(f, p, g, q)
end

network = Network()

function save(filename)

    f = getfield(network, :f)
    g = getfield(network, :g)
    @save filename f g
end

function load(filename)

    @load filename f g
    p = params(f)
    q = params(g)
    Flux.loadparams!(network.f, p)
    Flux.loadparams!(network.g, q)
end

function init()

    # layers1 parameters init
    parameters1 = Vector{Parameters}(undef, Const.layers1_num)
    for i in 1:Const.layers1_num-1
        W = Flux.glorot_normal(Const.layer1[i+1], Const.layer1[i])
        b = zeros(Float32, Const.layer1[i+1])
        parameters1[i] = Parameters(W, b)
    end
    W = randn(Complex{Float32}, Const.layer1[end], Const.layer1[end-1]) ./ sqrt(Const.layer1[end-1])
    b = zeros(Complex{Float32}, Const.layer1[end])
    parameters1[end] = Parameters(W, b)
    p = params([[parameters1[i].W, parameters1[i].b] for i in 1:Const.layers1_num]...)

    # layers2 parameters init
    parameters2 = Vector{Parameters}(undef, Const.layers2_num)
    for i in 1:Const.layers2_num-1
        W = Flux.glorot_normal(Const.layer2[i+1], Const.layer2[i])
        b = zeros(Float32, Const.layer2[i+1])
        parameters2[i] = Parameters(W, b)
    end
    W = randn(Complex{Float32}, Const.layer2[end], Const.layer2[end-1]) ./ sqrt(Const.layer2[end-1])
    b = zeros(Complex{Float32}, Const.layer2[end])
    parameters2[end] = Parameters(W, b)
    q = params([[parameters2[i].W, parameters2[i].b] for i in 1:Const.layers2_num]...)

    # Network initialize
    Flux.loadparams!(network.f, p)
    Flux.loadparams!(network.g, q)
end

function forward1(n::Vector{Float32})

    out = network.f(n)
    return out
end

function forward2(s::Vector{Float32})

    out = network.g(s)
    return out
end

loss(s::Vector{Float32}, n::Vector{Float32}) = real(transpose(forward2(s)) * forward1(n))

function backward(s::Vector{Float32}, n::Vector{Float32}, e::Complex{Float32})

    # Network1 Backward
    gs1 = gradient(() -> loss(s, n), network.p)
    for i in 1:Const.layers1_num-1
        dw = gs1[network.f[i].W]
        db = gs1[network.f[i].b]
        o1[i].W  += dw
        o1e[i].W += dw * e
        o1[i].b  += db
        o1e[i].b += db * e
    end
    dw = gs1[network.f[end].W]
    o1[end].W  += dw
    o1e[end].W += dw * e

    # Network2 Backward
    gs2 = gradient(() -> loss(s, n), network.q)
    for i in 1:Const.layers2_num-1
        dw = gs2[network.g[i].W]
        db = gs2[network.g[i].b]
        o2[i].W  += dw
        o2e[i].W += dw * e
        o2[i].b  += db
        o2e[i].b += db * e
    end
    dw = gs2[network.g[end].W]
    o2[end].W  += dw
    o2e[end].W += dw * e
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energyS::Float32, energyB::Float32, ϵ::Float32, lr::Float32)

    energy = energyS + energyB
    α = ifelse(lr > 0f0, 4.0f0 * (energy - ϵ), 1f0 * (energyB < 0f0)) / Const.iters_num

    # Newtork1 Update
    for i in 1:Const.layers1_num-1
        ΔW = α .* real.(o1e[i].W .- energy * o1[i].W)
        Δb = α .* real.(o1e[i].b .- energy * o1[i].b)
        update!(opt(lr), network.f[i].W, ΔW, o1[i].W)
        update!(opt(lr), network.f[i].b, Δb, o1[i].b)
    end
    ΔW = α .* (o1e[end].W .- energy * o1[end].W)
    Δb = α .* (o1e[end].b .- energy * o1[end].b)
    update!(opt(lr), network.f[end].W, ΔW, o1[end].W)

    # Newtork2 Update
    for i in 1:Const.layers2_num-1
        ΔW = α .* real.(o2e[i].W .- energy * o2[i].W)
        Δb = α .* real.(o2e[i].b .- energy * o2[i].b)
        update!(opt(lr), network.g[i].W, ΔW, o2[i].W)
        update!(opt(lr), network.g[i].b, Δb, o2[i].b)
    end
    ΔW = α .* (o2e[end].W .- energy * o2[end].W)
    Δb = α .* (o2e[end].b .- energy * o2[end].b)
    update!(opt(lr), network.g[end].W, ΔW, o2[end].W)
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
