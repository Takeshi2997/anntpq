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
    p  = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        p[i]  = Params(W, b)
    end
    ϕ = WaveFunction(0f0im, 0f0im)
    ParamSet(p, p, p, ϕ)
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
        dw = realgs[network.gX[i].W] - im * imaggs[network.gY[i].W]
        db = realgs[network.gX[i].b] - im * imaggs[network.gY[i].b]
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
    paramset.ϕ.x /= Const.iters_num
    X = 1f0 / sqrt(real(paramset.ϕ.x))
    ϕ =  X * paramset.ϕ.y / Const.iters_num
    for i in 1:Const.layers_num
        oW  = paramset.oX[i].W  / Const.iters_num
        ob  = paramset.oX[i].b  / Const.iters_num
        oeW = paramset.oXe[i].W / Const.iters_num
        oeb = paramset.oXe[i].b / Const.iters_num
        oϕW = X * paramset.oXϕ[i].W / Const.iters_num
        oϕb = X * paramset.oXϕ[i].b / Const.iters_num
        realΔW =  real.(oeW) - e * real.(oW) - real.(oϕW) + real.(oW) .* real(ϕ)
        imagΔW = -imag.(oeW) + e * imag.(oW) + imag.(oϕW) - imag.(oW) .* real(ϕ)
        realΔb =  real.(oeb) - e * real.(ob) - real.(oϕb) + real.(ob) .* real(ϕ) 
        imagΔb = -imag.(oeb) + e * imag.(ob) + imag.(oϕb) - imag.(ob) .* real(ϕ) 
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
