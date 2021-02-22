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
    oX::Vector{T}
    oXe::Vector{T}
    oXϕ::Vector{T}
    oY::Vector{T}
    oYe::Vector{T}
    oYϕ::Vector{T}
    ϕ::T
end

function ParamSet()
    p  = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        p[i]  = Params(W, b)
    end
    ϕ = WaveFunction(0f0im, 0f0im)
    ParamSet(p, p, p, p, p, p, ϕ)
end

# Define Network

mutable struct Network
    fX::Flux.Chain
    fY::Flux.Chain
    gX::Flux.Chain
    gY::Flux.Chain
    pX::Zygote.Params
    pY::Zygote.Params
    qX::Zygote.Params
    qY::Zygote.Params
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], swish)
    end
    layers[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network(f, f, f, f, p, p, p, p)
end

network = Network()

# Network Utility

function save(filename)
    fX = getfield(network, :fX)
    fY = getfield(network, :fY)
    @save filename fX fY
end

function load(filename)
    @load filename fX fY
    pX = Flux.params(fX)
    pY = Flux.params(fY)
    Flux.loadparams!(network.gX, pX)
    Flux.loadparams!(network.gY, pY)
end

function load_f(filename)
    @load filename fX fY
    pX = Flux.params(fX)
    pY = Flux.params(fY)
    Flux.loadparams!(network.fX, pX)
    Flux.loadparams!(network.fY, pY)
end

function reset()
    gX = getfield(network, :gX)
    gY = getfield(network, :gY)
    qX = Flux.params(gX)
    qY = Flux.params(gY)
    Flux.loadparams!(network.fX, qX)
    Flux.loadparams!(network.fY, qY)
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
    Flux.loadparams!(network.fX, p)
    Flux.loadparams!(network.fY, p)
end

# Learning Method

function forward(x::Vector{Float32})
    a = network.gX(x)[1]
    b = network.gY(x)[1]
    return a + im * b
end

function forward_f(x::Vector{Float32})
    a = network.fX(x)[1]
    b = network.fY(x)[1]
    return a + im * b
end

realloss(x::Vector{Float32}) = network.gX(x)[1]
imagloss(x::Vector{Float32}) = network.gY(x)[1]

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    realgs = gradient(() -> realloss(x), network.qX)
    imaggs = gradient(() -> imagloss(x), network.qY)
    ϕ = exp(forward_f(x) - forward(x))
    for i in 1:Const.layers_num
        realdw = realgs[network.gX[i].W]
        realdb = realgs[network.gX[i].b]
        imagdw = imaggs[network.gY[i].W]
        imagdb = imaggs[network.gY[i].b]
        paramset.oX[i].W  += realdw
        paramset.oX[i].b  += realdb
        paramset.oXe[i].W += realdw .* e
        paramset.oXe[i].b += realdb .* e
        paramset.oXϕ[i].W += realdw .* ϕ
        paramset.oXϕ[i].b += realdb .* ϕ
        paramset.oY[i].W  += imagdw
        paramset.oY[i].b  += imagdb
        paramset.oYe[i].W += imagdw .* e
        paramset.oYe[i].b += imagdb .* e
        paramset.oYϕ[i].W += imagdw .* ϕ
        paramset.oYϕ[i].b += imagdb .* ϕ
    end
    paramset.ϕ.x += conj(ϕ) * ϕ
    paramset.ϕ.y += ϕ
end

function updateparams(e::Float32, lr::Float32, paramset::ParamSet, Δparamset::Vector)
    paramset.ϕ.x /= Const.iters_num
    X = 1f0 / sqrt(real(paramset.ϕ.x))
    ϕ =  X * paramset.ϕ.y / Const.iters_num
    for i in 1:Const.layers_num
        oXW  = real.(paramset.oX[i].W  / Const.iters_num)
        oXb  = real.(paramset.oX[i].b  / Const.iters_num)
        oXeW = paramset.oXe[i].W / Const.iters_num
        oXeb = paramset.oXe[i].b / Const.iters_num
        oXϕW = X * paramset.oXϕ[i].W / Const.iters_num
        oXϕb = X * paramset.oXϕ[i].b / Const.iters_num
        oYW  = real.(paramset.oY[i].W  / Const.iters_num)
        oYb  = real.(paramset.oY[i].b  / Const.iters_num)
        oYeW = paramset.oYe[i].W / Const.iters_num
        oYeb = paramset.oYe[i].b / Const.iters_num
        oYϕW = X * paramset.oYϕ[i].W / Const.iters_num
        oYϕb = X * paramset.oYϕ[i].b / Const.iters_num
        realΔW = real.(oXeW) - e * oXW - real.(oXϕW) + oXW .* real(ϕ) + imag.(oYeW)  - imag.(oYϕW) + oYW .* imag(ϕ)
        imagΔW = imag.(oXeW) - imag.(oXϕW) + oXW .* imag(ϕ) - real.(oYeW) + e .* oYW + real.(oYϕW) - oYW .* real(ϕ)
        realΔb = real.(oXeb) - e * oXb - real.(oXϕb) + oXb .* real(ϕ) + imag.(oYeb)  - imag.(oYϕb) + oYb .* imag(ϕ)
        imagΔb = imag.(oXeb) - imag.(oXϕb) + oXb .* imag(ϕ) - real.(oYeb) + e .* oYb + real.(oYϕb) - oYb .* real(ϕ)
        Δparamset[i][1] += realΔW
        Δparamset[i][2] += imagΔW
        Δparamset[i][3] += realΔb
        Δparamset[i][4] += imagΔb
    end
end

opt(lr::Float32) = AMSGrad(lr, (0.9, 0.999))

function update(Δparamset::Vector, lr::Float32)
    for i in 1:Const.layers_num
        ΔrealW = Δparamset[i][1]
        ΔimagW = Δparamset[i][2]
        Δrealb = Δparamset[i][3]
        Δimagb = Δparamset[i][4]
        update!(opt(lr), network.gX[i].W, ΔrealW)
        update!(opt(lr), network.gY[i].W, ΔimagW)
        update!(opt(lr), network.gX[i].b, Δrealb)
        update!(opt(lr), network.gY[i].b, Δimagb)
    end
end
end
