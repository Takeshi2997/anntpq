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
    o::Vector{Vector{T}}
    oe::Vector{Vector{T}}
    oϕ::Vector{Vector{T}}
    ϕ::T
end

function ParamSet()
    p  = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        p[i]  = Params(W, b)
    end
    pvec = [p, p]
    ϕ = WaveFunction(0f0im, 0f0im)
    ParamSet(pvec, pvec, pvec, ϕ)
end

# Define Network

mutable struct Network
    f::Vector{Flux.Chain}
    g::Vector{Flux.Chain}
    p::Vector{Zygote.Params}
    q::Vector{Zygote.Params}
    λ::Float32
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], swish)
    end
    layers[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network([f, f], [f, f], [p, p], [p, p], 0f0)
end

network = Network()

# Network Utility

function save(filename)
    f = getfield(network, :f)
    f1 = f[1]
    f2 = f[2]
    @save filename f1 f2
end

function load(filename)
    @load filename f1 f2
    p1 = Flux.params(f1)
    p2 = Flux.params(f2)
    Flux.loadparams!(network.g[1], p1)
    Flux.loadparams!(network.g[2], p2)
end

function load_f(filename)
    @load filename f1 f2
    p1 = Flux.params(f1)
    p2 = Flux.params(f2)
    Flux.loadparams!(network.f[1], p1)
    Flux.loadparams!(network.f[2], p2)
end

function reset()
    g = getfield(network, :g)
    q1 = Flux.params(g[1])
    q2 = Flux.params(g[2])
    Flux.loadparams!(network.f[1], q1)
    Flux.loadparams!(network.f[2], q2)
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
    Flux.loadparams!(network.f[1], p)
    Flux.loadparams!(network.f[2], p)
end

# Learning Method

function forward(x::Vector{Float32})
    a = network.g[1](x)[1]
    b = network.g[2](x)[1]
    return a + im * b
end

function forward_f(x::Vector{Float32})
    a = network.f[1](x)[1]
    b = network.f[2](x)[1]
    return a + im * b
end

realloss(x::Vector{Float32}) = network.g[1](x)[1]
imagloss(x::Vector{Float32}) = network.g[2](x)[1]

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    realgs = gradient(() -> realloss(x), network.q[1])
    imaggs = gradient(() -> imagloss(x), network.q[2])
    ϕ = exp(forward_f(x) - forward(x))
    for i in 1:Const.layers_num
        dxw = realgs[network.g[1][i].W]
        dyw = imaggs[network.g[2][i].W]
        dxb = realgs[network.g[1][i].b]
        dyb = imaggs[network.g[2][i].b]
        paramset.o[1][i].W  += dxw
        paramset.o[1][i].b  += dxb
        paramset.oe[1][i].W += dxw .* e
        paramset.oe[1][i].b += dxb .* e
        paramset.oϕ[1][i].W += dxw .* ϕ
        paramset.oϕ[1][i].b += dxb .* ϕ
        paramset.oe[2][i].W += dyw .* e
        paramset.oe[2][i].b += dyb .* e
        paramset.oϕ[2][i].W += dyw .* ϕ
        paramset.oϕ[2][i].b += dyb .* ϕ
    end
    paramset.ϕ.x += conj(ϕ) * ϕ
    paramset.ϕ.y += ϕ
end

function updateparams(energy::Float32, ϵ::Float32, ϕ::Float32, paramset::ParamSet, Δparamset::Vector)
    paramset.ϕ.x /= Const.iters_num
    X = 1f0 / sqrt(real(paramset.ϕ.x))
    ϕ = real(paramset.ϕ.y / Const.iters_num * X)
    λ = network.λ
    for i in 1:Const.layers_num
        oWx   = real.(paramset.o[1][i].W  / Const.iters_num)
        obx   = real.(paramset.o[1][i].b  / Const.iters_num)
        oeWx  = real.(paramset.oe[1][i].W / Const.iters_num)
        oebx  = real.(paramset.oe[1][i].b / Const.iters_num)
        oϕWx  = real.(paramset.oϕ[1][i].W / Const.iters_num .* X)
        oϕbx  = real.(paramset.oϕ[1][i].b / Const.iters_num .* X)
        oeWy  = imag.(paramset.oe[2][i].W / Const.iters_num)
        oeby  = imag.(paramset.oe[2][i].b / Const.iters_num)
        oϕWy  = imag.(paramset.oϕ[2][i].W / Const.iters_num .* X)
        oϕby  = imag.(paramset.oϕ[2][i].b / Const.iters_num .* X)
        realΔW = λ .* (energy - ϵ) .* (oeWx - energy * oWx) - oϕWx + oWx .* ϕ
        realΔb = λ .* (energy - ϵ) .* (oebx - energy * obx) - oϕbx + obx .* ϕ
        imagΔW = λ .* (energy - ϵ) .*  oeWy - oϕWy
        imagΔb = λ .* (energy - ϵ) .*  oeby - oϕby
        Δparamset[i][1] += realΔW
        Δparamset[i][2] += imagΔW
        Δparamset[i][3] += realΔb
        Δparamset[i][4] += imagΔb
    end
end

opt(lr::Float32) = Descent(lr)

function update(energy::Float32, ϵ::Float32, Δparamset::Vector, lr::Float32)
    for i in 1:Const.layers_num
        update!(opt(lr), network.g[1][i].W, Δparamset[i][1])
        update!(opt(lr), network.g[2][i].W, Δparamset[i][2])
        update!(opt(lr), network.g[1][i].b, Δparamset[i][3])
        update!(opt(lr), network.g[2][i].b, Δparamset[i][4])
    end
    λ = network.λ - lr * (energy - ϵ)^2 / 2f0
    setfield!(network, :λ, λ)
end
end
