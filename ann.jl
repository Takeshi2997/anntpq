module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux: @functor
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

# Initialize
abstract type Parameters end
mutable struct Params{S<:AbstractArray, T<:AbstractArray} <: Parameters
    W::S
    b::T
end
mutable struct WaveFunction{S<:Complex} <: Parameters
    x::S
    y::S
end

mutable struct ParamSet{T <: Parameters}
    o::Vector{T}
    oe::Vector{T}
    oϕ::Vector{T}
    ϕ::T
end

function ParamSet()
    o  = Vector{Parameters}(undef, Const.layers_num)
    oe = Vector{Parameters}(undef, Const.layers_num)
    oϕ = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        o[i]  = Params(W, b)
        oe[i] = Params(W, b)
        oϕ[i] = Params(W, b)
    end
    ϕ = WaveFunction(0f0im, 0f0im)
    ParamSet(o, oe, oϕ, ϕ)
end

# Define Network

mutable struct Network
    f::Flux.Chain
    g::Flux.Chain
    p::Zygote.Params
    q::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], swish)
    end
    layers[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, f, p, p)
end

network = Network()

# Network Utility

function save(filename)
    f = getfield(network, :f)
    @save filename f
end

function load(filename)
    @load filename f
    p = Flux.params(f)
    Flux.loadparams!(network.g, p)
end

function load_f(filename)
    @load filename f
    p = Flux.params(f)
    Flux.loadparams!(network.f, p)
end

function reset()
    g = getfield(network, :g)
    q = Flux.params(g)
    Flux.loadparams!(network.f, q)
end

function init_sub()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.kaiming_normal(Const.layer[i+1], Const.layer[i])
        b = Flux.zeros(Const.layer[i+1]) 
        parameters[i] = [W, b]
    end
    paramset = [param for param in parameters]
    q = Flux.params(paramset...)
    Flux.loadparams!(network.g, q)
end

function init()
    parameters = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.kaiming_normal(Const.layer[i+1], Const.layer[i])
        b = Flux.zeros(Const.layer[i+1])
        parameters[i] = [W, b]
    end
    paramset = [param for param in parameters]
    p = Flux.params(paramset...)
    Flux.loadparams!(network.f, p)
end

# Learning Method

function forward(x::Vector{Float32})
    out = network.g(x)
    return out[1] + im * out[2]
end

function forward_b(x::Vector{Float32})
    out = network.f(x)
    return out[1] + im * out[2]
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    gs = gradient(() -> loss(x), network.q)
    ϕ = exp(forward_b(x) - forward(x))
    for i in 1:Const.layers_num
        dw = gs[network.g[i].W]
        db = gs[network.g[i].b]
        paramset.o[i].W  += dw
        paramset.o[i].b  += db
        paramset.oe[i].W += dw .* e
        paramset.oe[i].b += db .* e
        paramset.oϕ[i].W += dw .* ϕ
        paramset.oϕ[i].b += db .* ϕ
    end
    paramset.ϕ.x += conj(ϕ) * ϕ
    paramset.ϕ.y += ϕ
end

function updateparams(e::Float32, lr::Float32, paramset::ParamSet, Δparamset::Vector)
    for i in 1:Const.layers_num
        paramset.o[i].W  ./= Const.iters_num
        paramset.o[i].b  ./= Const.iters_num
        paramset.oe[i].W ./= Const.iters_num
        paramset.oe[i].b ./= Const.iters_num
        paramset.oϕ[i].W ./= Const.iters_num
        paramset.oϕ[i].b ./= Const.iters_num
    end
    paramset.ϕ.x /= Const.iters_num
    paramset.ϕ.y /= Const.iters_num
    X = 1f0 / sqrt(real(paramset.ϕ.x))
    u = e / 2f0 - real(X * paramset.ϕ.y)
    v = imag(X * paramset.ϕ.y)
    r = sqrt(u^2 + v^2)
    for i in 1:Const.layers_num
        ∂uW = real.(paramset.oe[i].W - e * paramset.o[i].W) -
              X * (real.(paramset.oϕ[i].W) - real.(paramset.o[i].W) .* real.(paramset.ϕ.y))
        ∂vW = X * (imag.(paramset.oϕ[i].W) - real.(paramset.o[i].W) .* imag.(paramset.ϕ.y))
        Δparamset[i][1] += (u .* ∂uW + v * ∂vW) / r
        ∂ub = real.(paramset.oe[i].b - e * paramset.o[i].b) -
              X * (real.(paramset.oϕ[i].b) - real.(paramset.o[i].b) .* real.(paramset.ϕ.y))
        ∂vb = X * (imag.(paramset.oϕ[i].b) - real.(paramset.o[i].b) .* imag.(paramset.ϕ.y))
        Δparamset[i][2] += (u .* ∂ub + v .* ∂vb) / r
    end
end

opt1(lr::Float32) = Descent(lr)
opt2(lr::Float32) = AMSGrad(lr, (0.9, 0.999))

function update(Δparamset::Vector, lr::Float32, n::Integer)
    opt = ifelse(n > 20, opt1, opt2)
    for i in 1:Const.layers_num
        ΔW = Δparamset[i][1]
        Δb = Δparamset[i][2]
        update!(opt(lr), network.g[i].W, ΔW)
        update!(opt(lr), network.g[i].b, Δb)
    end
end
end
