module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, BlockDiagonals
using Flux: @functor
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array
    b::Array
end

mutable struct Outputparams

    W::Array
    b::Array
    a::Array
end

o  = Vector{Any}(undef, Const.layers_num)
oe = Vector{Any}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num-1
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Parameters(W, b)
        global oe[i] = Parameters(W, b)
    end
    W = zeros(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    a = zeros(Complex{Float32}, Const.layer[1])
    global o[end]  = Outputparams(W, b, a)
    global oe[end] = Outputparams(W, b, a)
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

struct Output{F,S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
    a::T
    σ::F
end

@functor Output

function (m::Output)(x::AbstractArray)
  W, b, a, σ = m.W, m.b, m.a, m.σ
  σ.(W*x.+b), a
end

NNlib.logcosh(x::Complex{Float32}) = log(cosh(x))

function Network()

    layer = Vector{Any}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], logcosh)
    end
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    a = randn(Complex{Float32}, Const.layer[1])
    layer[end] = Output(W, b, a, logcosh)
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

    parameters = Vector{Any}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        W = randn(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = Parameters(W, b)
    end
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1])
    b = zeros(Complex{Float32}, Const.layer[end])
    a = randn(Complex{Float32}, Const.layer[1])
    parameters[end] = Outputparams(W, b, a)
    paramset = [param for param in parameters]
    p = params(paramset...)
    Flux.loadparams!(network.f, p)
end

function forward(x::Vector{Float32})

    main, bias = network.f(x)
    return sum(main) + transpose(x) * bias
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
    db = gs[network.f[end].b]
    da = gs[network.f[end].a]
    o[end].W  += dw
    oe[end].W += dw * e
    o[end].b  += db
    oe[end].b += db * e
    o[end].a  += da
    oe[end].a += da * e
end

opt(lr::Float32) = QRMSProp(lr, 0.9)

function update(energy::Float32, ϵ::Float32, lr::Float32)

    α = 4.0f0 * (energy - ϵ) / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW, o[i].W)
        update!(opt(lr), network.f[i].b, Δb, o[i].b)
    end
    ΔW = α .* oe[end].W .- energy * o[end].W
    Δb = α .* oe[end].b .- energy * o[end].b
    Δa = α .* oe[end].a .- energy * o[end].a
    update!(opt(lr), network.f[end].W, ΔW, o[end].W)
    update!(opt(lr), network.f[end].b, Δb, o[end].b)
    update!(opt(lr), network.f[end].a, Δa, o[end].a)
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
