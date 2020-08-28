module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote, BlockDiagonals
using Flux: @functor
using Flux.Optimise: update!
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
        layer[i] = Dense(Const.layer[i], Const.layer[i+1], tanh)
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

    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        W = randn(Float32, Const.layer[i+1], Const.layer[i]) ./ Float32(sqrt(Const.layer[i]))
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = [W, b]
    end
    W = randn(Complex{Float32}, Const.layer[end], Const.layer[end-1]) ./ Float32(sqrt(Const.layer[end-1]))
    b = zeros(Complex{Float32}, Const.layer[end])
    a = randn(Complex{Float32}, Const.layer[1]) ./ Float32(sqrt(Const.layer[1]))
    parameters[end] = [W, b, a]
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

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

const η = 0.05

function update(energy::Float32, ϵ::Float32, lr::Float32)

    α = (1f0 * (energy - ϵ > η) - 1f0 * (energy - ϵ < -η)) / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* 2f0 .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* 2f0 .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = α .* (oe[end].W .- energy * o[end].W)
    Δb = α .* (oe[end].b .- energy * o[end].b)
    Δa = α .* (oe[end].a .- energy * o[end].a)
    update!(opt(lr), network.f[end].W, ΔW)
    update!(opt(lr), network.f[end].b, Δb)
    update!(opt(lr), network.f[end].a, Δa)
end

end
