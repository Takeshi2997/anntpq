module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array
    b::Array
end

o   = Vector{Any}(undef, Const.layers_num)
oe  = Vector{Any}(undef, Const.layers_num)
oI  = Parameters
oIe = Parameters

function initO()

    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Parameters(W, b)
        global oe[i] = Parameters(W, b)
    end
    W = zeros(Complex{Float32}, Const.layer[1], Const.layer[1])
    b = zeros(Complex{Float32}, Const.layer[1])
    global oI  = Parameters(W, b)
    global oIe = Parameters(W, b)
end

mutable struct Network{F,P<:Zygote.Params}

    f::F
    p::P
end

struct Res{F,S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
    σ::F
end

function Res(in::Integer, out::Integer, σ = identity;
             initW = Flux.glorot_uniform, initb = zeros)
    return Res(initW(out, in), initb(Float32, out), σ)
end

@functor Res

function (m::Res)(x::AbstractArray)
    W, b, σ = m.W, m.b, m.σ
    x .+ σ.(W*x.+b)
end

function Network()

    layer = Vector{Any}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Res(Const.layer[i], Const.layer[i+1], tanh)
    end
    layer[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = params(f)
    Network(f, p)
end

function Affine()

    W = randn(Complex{Float32}, Const.layer[1], Const.layer[1])
    b = zeros(Complex{Float32}, Const.layer[1])
    f = Dense(W, b)
    p = params(f)
    Network(f, p)
end

network = Network()
affineI = Affine()

function save(filename)

    f = getfield(network, :f)
    g = getfield(affineI, :f)
    @save filename f g
end

function load(filename)

    @load filename f g
    p = params(f)
    q = params(g)
    Flux.loadparams!(network.f, p)
    Flux.loadparams!(affineI.f, q)
end

function init()

    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.glorot_normal(Const.layer[i+1], Const.layer[i])
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = [W, b]
    end
    paramset = [param for param in parameters]
    p = params(paramset...)
    W = randn(Complex{Float32}, Const.layer[1], Const.layer[1]) ./ Float32(Const.layer[1])
    b = zeros(Complex{Float32}, Const.layer[1])
    q = params([W, b])
    Flux.loadparams!(network.f, p)
    Flux.loadparams!(affineI.f, q)
end

function forward(α::Vector{Float32})

    out = network.f(α)
    return out[1] + im * out[2]
end

function interaction(α::Vector{Float32})

    return affineI.f(α)
end

function loss(x::Vector{Float32}, α::Vector{Float32})

    out = network.f(α)
    return real(out[1] + im * out[2] + dot(x, interaction(α)))
end

function backward(x::Vector{Float32}, α::Vector{Float32}, e::Complex{Float32})

    gs = gradient(() -> loss(x, α), network.p)
    for i in 1:Const.layers_num
        dw = gs[network.f[i].W]
        db = gs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
    dw = x .* transpose(α)
    db = x
    oI.W  += dw
    oIe.W += dw * e
    oI.b  += db
    oIe.b += db * e
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)

    x = (energy - ϵ)
    α = 2f0 * x / Const.iters_num
    for i in 1:Const.layers_num
        ΔW = α .* 2f0 .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* 2f0 .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = α .* (oIe.W .- energy * oI.W)
    Δb = α .* (oIe.b .- energy * oI.b)
    update!(opt(lr), affineI.f.W, ΔW)
    update!(opt(lr), affineI.f.b, Δb)
end

end
