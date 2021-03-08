module ANN
include("../setup.jl")
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

mutable struct ParamSet{T <: Parameters}
    o::Vector{Vector{T}}
    oe::Vector{Vector{T}}
end

function ParamSet()
    p  = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i])
        b = zeros(Complex{Float32}, Const.layer[i+1])
        p[i]  = Params(W, b)
    end
    pvec = [p, p]
    ParamSet(pvec, pvec)
end

# Define Network

mutable struct Network
    f::Vector{Flux.Chain}
    p::Vector{Zygote.Params}
end

function Network()
    layers = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers[i] = Dense(Const.layer[i], Const.layer[i+1], swish)
    end
    layers[end] = Dense(Const.layer[end-1], Const.layer[end])
    f = Chain([layers[i] for i in 1:Const.layers_num]...)
    p = Flux.params(f)
    Network([f, f], [p, p])
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
    Flux.loadparams!(network.f[1], p1)
    Flux.loadparams!(network.f[2], p2)
end

# Learning Method

function forward(x::Vector{Float32})
    a = network.f[1](x)[1]
    b = network.f[2](x)[1]
    return a + im * b
end

realloss(x::Vector{Float32}) = network.f[1](x)[1]
imagloss(x::Vector{Float32}) = network.f[2](x)[1]

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    realgs = gradient(() -> realloss(x), network.p[1])
    imaggs = gradient(() -> imagloss(x), network.p[2])
    for i in 1:Const.layers_num
        dxw = realgs[network.f[1][i].W]
        dyw = imaggs[network.f[2][i].W]
        dxb = realgs[network.f[1][i].b]
        dyb = imaggs[network.f[2][i].b]
        paramset.o[1][i].W  += dxw
        paramset.o[1][i].b  += dxb
        paramset.oe[1][i].W += dxw .* e
        paramset.oe[1][i].b += dxb .* e
        paramset.oe[2][i].W += dyw .* e
        paramset.oe[2][i].b += dyb .* e
    end
end

function updateparams(e::Float32, paramset::ParamSet, Δparamset::Vector)
    for i in 1:Const.layers_num
        oWx   = real.(paramset.o[1][i].W  / Const.iters_num)
        obx   = real.(paramset.o[1][i].b  / Const.iters_num)
        oeWx  = real.(paramset.oe[1][i].W / Const.iters_num)
        oebx  = real.(paramset.oe[1][i].b / Const.iters_num)
        oeWy  = imag.(paramset.oe[2][i].W / Const.iters_num)
        oeby  = imag.(paramset.oe[2][i].b / Const.iters_num)
        realΔW = oeWx - e * oWx
        realΔb = oebx - e * obx
        imagΔW = oeWy
        imagΔb = oeby
        Δparamset[i][1] += realΔW
        Δparamset[i][2] += imagΔW
        Δparamset[i][3] += realΔb
        Δparamset[i][4] += imagΔb
    end
end

opt(lr::Float32) = AMSGrad(lr, (0.9, 0.999))

function update(Δparamset::Vector, lr::Float32)
    for i in 1:Const.layers_num
        update!(opt(lr), network.g[1][i].W, Δparamset[i][1])
        update!(opt(lr), network.g[2][i].W, Δparamset[i][2])
        update!(opt(lr), network.g[1][i].b, Δparamset[i][3])
        update!(opt(lr), network.g[2][i].b, Δparamset[i][4])
    end
end
end
