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
    realo::Vector{T}
    realoe::Vector{T}
    realoϕ::Vector{T}
    imago::Vector{T}
    imagoϕ::Vector{T}
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
    ParamSet(p, p, p, p, p, ϕ)
end

# Define Network

mutable struct Network
    realf::Flux.Chain
    imagf::Flux.Chain
    realg::Flux.Chain
    imagg::Flux.Chain
    realp::Zygote.Params
    imagp::Zygote.Params
    realq::Zygote.Params
    imagq::Zygote.Params
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
    realf = getfield(network, :realf)
    imagf = getfield(network, :imagf)
    @save filename realf imagf
end

function load(filename)
    @load filename realf imagf
    realp = Flux.params(realf)
    imagp = Flux.params(imagf)
    Flux.loadparams!(network.realg, realp)
    Flux.loadparams!(network.imagg, imagp)
end

function load_f(filename)
    @load filename realf imagf
    realp = Flux.params(realf)
    imagp = Flux.params(imagf)
    Flux.loadparams!(network.realf, realp)
    Flux.loadparams!(network.imagf, imagp)
end

function reset()
    realg = getfield(network, :realg)
    imagg = getfield(network, :imagg)
    realq = Flux.params(realg)
    imagq = Flux.params(imagg)
    Flux.loadparams!(network.realf, realq)
    Flux.loadparams!(network.imagf, imagq)
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
    Flux.loadparams!(network.realf, p)
    Flux.loadparams!(network.imagf, p)
end

# Learning Method

function forward(x::Vector{Float32})
    realout = network.realg(x)[1]
    imagout = network.imagg(x)[1]
    return realout + im * imagout
end

function forward_f(x::Vector{Float32})
    realout = network.realf(x)[1]
    imagout = network.imagf(x)[1]
    return realout + im * imagout
end

realloss(x::Vector{Float32}) = network.realg(x)[1]
imagloss(x::Vector{Float32}) = network.imagg(x)[1]

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    realgs = gradient(() -> realloss(x), network.realq)
    imaggs = gradient(() -> imagloss(x), network.imagq)
    ϕ = exp(forward_f(x) - forward(x))
    for i in 1:Const.layers_num
        realdw = realgs[network.realg[i].W]
        realdb = realgs[network.realg[i].b]
        imagdw = imaggs[network.imagg[i].W]
        imagdb = imaggs[network.imagg[i].b]
        paramset.realo[i].W  += realdw
        paramset.realo[i].b  += realdb
        paramset.realoe[i].W += realdw .* e
        paramset.realoe[i].b += realdb .* e
        paramset.realoϕ[i].W += realdw .* ϕ
        paramset.realoϕ[i].b += realdb .* ϕ
        paramset.imago[i].W  += imagdw
        paramset.imago[i].b  += imagdb
        paramset.imagoϕ[i].W += imagdw .* ϕ
        paramset.imagoϕ[i].b += imagdb .* ϕ
    end
    paramset.ϕ.x += conj(ϕ) * ϕ
    paramset.ϕ.y += ϕ
end

function updateparams(e::Float32, lr::Float32, paramset::ParamSet, Δparamset::Vector)
    for i in 1:Const.layers_num
        paramset.realo[i].W  ./= Const.iters_num
        paramset.realo[i].b  ./= Const.iters_num
        paramset.realoe[i].W ./= Const.iters_num
        paramset.realoe[i].b ./= Const.iters_num
        paramset.realoϕ[i].W ./= Const.iters_num
        paramset.realoϕ[i].b ./= Const.iters_num
        paramset.imago[i].W  ./= Const.iters_num
        paramset.imago[i].b  ./= Const.iters_num
        paramset.imagoϕ[i].W ./= Const.iters_num
        paramset.imagoϕ[i].b ./= Const.iters_num
    end
    paramset.ϕ.x /= Const.iters_num
    paramset.ϕ.y /= Const.iters_num
    X = 1f0 / sqrt(real(paramset.ϕ.x))
    for i in 1:Const.layers_num
        realΔW = real.(paramset.realoe[i].W - e * paramset.realo[i].W) -
                  X * (real.(paramset.realoϕ[i].W) - real.(paramset.realo[i].W) .* real.(paramset.ϕ.y))
        imagΔW = -X * (imag.(paramset.imagoϕ[i].W) - real.(paramset.imago[i].W) .* imag.(paramset.ϕ.y))
        Δparamset[i][1] += realΔW
        Δparamset[i][2] += imagΔW
        realΔb = real.(paramset.realoe[i].b - e * paramset.realo[i].b) -
                  X * (real.(paramset.realoϕ[i].b) - real.(paramset.realo[i].b) .* real.(paramset.ϕ.y))
        imagΔb = -X * (imag.(paramset.imagoϕ[i].b) - real.(paramset.realo[i].b) .* imag.(paramset.ϕ.y))
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
        update!(opt(lr), network.realg[i].W, ΔrealW)
        update!(opt(lr), network.imagg[i].W, ΔimagW)
        update!(opt(lr), network.realg[i].b, Δrealb)
        update!(opt(lr), network.imagg[i].b, Δimagb)
    end
end
end
