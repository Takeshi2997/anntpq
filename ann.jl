module ANN
include("./setup.jl")
include("./legendreTF.jl")
using .Const, .LegendreTF, LinearAlgebra, Flux, Zygote, Distributions
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array
    b::Array
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        global o[i]  = Parameters(W, b)
        global oe[i] = Parameters(W, b)
    end
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

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    layer = Vector{Any}(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layer[i] = Res(Const.layer[i], Const.layer[i+1], hardtanh)
    end
    layer[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layer[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, p)
end

network = Network()

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
        b = zeros(Float32, Const.layer[i+1])
        parameters[i] = [W, b]
    end
    e = Exponential(2f0)
    W = Array{Float32, 2}(undef, Const.layer[end], Const.layer[end-1])
    W[1, :] = log.(sqrt.(rand(e, Const.layer[end-1])))
    W[2, :] = rand(Float32, Const.layer[end-1]) .* 2f0 .* π .- π
    b = zeros(Float32, Const.layer[end])
    parameters[end] = [W, b]
 
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
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

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)

    x = (energy - ϵ)
    α = 2f0 * x / Const.iters_num
    for i in 1:Const.layers_num-1
        ΔW = α .* 2f0 .* real.(oe[i].W .- energy * o[i].W)
        Δb = α .* 2f0 .* real.(oe[i].b .- energy * o[i].b)
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = α .* 2f0 * real(oe[end].W .- energy * o[end].W)
    update!(opt(lr), network.f[end].W, ΔW)
end

end
