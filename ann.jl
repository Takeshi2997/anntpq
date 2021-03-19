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

mutable struct ParamSet{T <: Parameters}
    o::Vector{Vector{T}}
    oe::Vector{Vector{T}}
end

function ParamSet()
    p1  = Vector{Parameters}(undef, Const.layers_num)
    p2  = Vector{Parameters}(undef, Const.layers_num)
    p3  = Vector{Parameters}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W1 = zeros(Complex{Float32}, Const.layer1[i+1], Const.layer1[i])
        b1 = zeros(Complex{Float32}, Const.layer1[i+1])
        W2 = zeros(Complex{Float32}, Const.layer2[i+1], Const.layer2[i])
        b2 = zeros(Complex{Float32}, Const.layer2[i+1])
        W3 = zeros(Complex{Float32}, Const.layer3[i+1], Const.layer3[i])
        b3 = zeros(Complex{Float32}, Const.layer3[i+1])
        p1[i]  = Params(W1, b1)
        p2[i]  = Params(W2, b2)
        p3[i]  = Params(W3, b3)
    end
    ParamSet([p1, p2, p3], [p1, p2, p3])
end

# Define Network

mutable struct Network
    f::Vector{Flux.Chain}
    p::Vector{Zygote.Params}
end

function Network()
    layers1 = Vector(undef, Const.layers_num)
    layers2 = Vector(undef, Const.layers_num)
    layers3 = Vector(undef, Const.layers_num)
    for i in 1:Const.layers_num-1
        layers1[i] = Dense(Const.layer1[i], Const.layer1[i+1], swish)
    end
    layers1[end] = Dense(Const.layer1[end-1], Const.layer1[end])
    for i in 1:Const.layers_num-1
        layers2[i] = Dense(Const.layer2[i], Const.layer2[i+1], swish)
    end
    layers2[end] = Dense(Const.layer2[end-1], Const.layer2[end])
    for i in 1:Const.layers_num-1
        layers3[i] = Dense(Const.layer3[i], Const.layer3[i+1], swish)
    end
    layers3[end] = Dense(Const.layer3[end-1], Const.layer3[end])
    f1 = Chain([layers1[i] for i in 1:Const.layers_num]...)
    f2 = Chain([layers2[i] for i in 1:Const.layers_num]...)
    f3 = Chain([layers3[i] for i in 1:Const.layers_num]...)
    p1 = Flux.params(f1)
    p2 = Flux.params(f2)
    p3 = Flux.params(f3)
    Network([f1, f2, f3], [p1, p2, p3])
end

network = Network()

# Network Utility

function save(filename)
    f = getfield(network, :f)
    f1 = f[1]
    f2 = f[2]
    f3 = f[3]
    @save filename f1 f2 f3
end

function load(filename)
    @load filename f1 f2 f3
    p1 = Flux.params(f1)
    p2 = Flux.params(f2)
    p3 = Flux.params(f3)
    Flux.loadparams!(network.f[1], p1)
    Flux.loadparams!(network.f[2], p2)
    Flux.loadparams!(network.f[3], p3)
end

function init()
    parameters1 = Vector{Array}(undef, Const.layers_num)
    parameters2 = Vector{Array}(undef, Const.layers_num)
    parameters3 = Vector{Array}(undef, Const.layers_num)
    for i in 1:Const.layers_num
        W = Flux.kaiming_normal(Const.layer1[i+1], Const.layer1[i])
        b = Flux.zeros(Const.layer1[i+1])
        parameters1[i] = [W, b]
    end
    for i in 1:Const.layers_num
        W = Flux.kaiming_normal(Const.layer2[i+1], Const.layer2[i])
        b = Flux.zeros(Const.layer2[i+1])
        parameters2[i] = [W, b]
    end
    for i in 1:Const.layers_num
        W = Flux.kaiming_normal(Const.layer3[i+1], Const.layer3[i])
        b = Flux.zeros(Const.layer3[i+1])
        parameters3[i] = [W, b]
    end
    paramset1 = [param for param in parameters1]
    paramset2 = [param for param in parameters2]
    paramset3 = [param for param in parameters3]
    p1 = Flux.params(paramset1...)
    p2 = Flux.params(paramset2...)
    p3 = Flux.params(paramset3...)
    Flux.loadparams!(network.f[1], p1)
    Flux.loadparams!(network.f[2], p2)
    Flux.loadparams!(network.f[3], p3)
end

# Learning Method

function forward(x::Vector{Float32})
    out1 = network.f[1](@views x[1:Const.dimB])
    out2 = network.f[2](@views x[1+Const.dimB:end])
    out3 = network.f[3](x)
    return out1[1] + im * out1[2] + out2[1] + im * out2[2] + Const.η * (out3[1] + im * out3[2])
end

loss(x::Vector{Float32}) = real(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32}, paramset::ParamSet)
    gs1 = gradient(() -> loss(x), network.p[1])
    gs2 = gradient(() -> loss(x), network.p[2])
    gs3 = gradient(() -> loss(x), network.p[3])
    for i in 1:Const.layers_num
        dw1 = gs1[network.f[1][i].W]
        db1 = gs1[network.f[1][i].b]
        dw2 = gs2[network.f[2][i].W]
        db2 = gs2[network.f[2][i].b]
        dw3 = gs3[network.f[3][i].W]
        db3 = gs3[network.f[3][i].b]
        paramset.o[1][i].W  += dw1
        paramset.o[1][i].b  += db1
        paramset.oe[1][i].W += dw1 .* e
        paramset.oe[1][i].b += db1 .* e
        paramset.o[2][i].W  += dw2
        paramset.o[2][i].b  += db2
        paramset.oe[2][i].W += dw2 .* e
        paramset.oe[2][i].b += db2 .* e
        paramset.o[3][i].W  += dw3
        paramset.o[3][i].b  += db3
        paramset.oe[3][i].W += dw3 .* e
        paramset.oe[3][i].b += db3 .* e
    end
end

function updateparams(energy::Float32, paramset::ParamSet, Δparamset::Vector)
    for i in 1:Const.layers_num
        o1W   = real.(paramset.o[1][i].W  / Const.iters_num)
        o1b   = real.(paramset.o[1][i].b  / Const.iters_num)
        oe1W  = real.(paramset.oe[1][i].W / Const.iters_num)
        oe1b  = real.(paramset.oe[1][i].b / Const.iters_num)
        o2W   = real.(paramset.o[2][i].W  / Const.iters_num)
        o2b   = real.(paramset.o[2][i].b  / Const.iters_num)
        oe2W  = real.(paramset.oe[2][i].W / Const.iters_num)
        oe2b  = real.(paramset.oe[2][i].b / Const.iters_num)
        o3W   = real.(paramset.o[3][i].W  / Const.iters_num)
        o3b   = real.(paramset.o[3][i].b  / Const.iters_num)
        oe3W  = real.(paramset.oe[3][i].W / Const.iters_num)
        oe3b  = real.(paramset.oe[3][i].b / Const.iters_num)
        ΔW1 = oe1W - energy * o1W
        Δb1 = oe1b - energy * o1b
        ΔW2 = oe2W - energy * o2W
        Δb2 = oe2b - energy * o2b
        ΔW3 = oe3W - energy * o3W
        Δb3 = oe3b - energy * o3b
        Δparamset[i][1] += ΔW1
        Δparamset[i][2] += Δb1
        Δparamset[i][3] += ΔW2
        Δparamset[i][4] += Δb2
        Δparamset[i][5] += ΔW3
        Δparamset[i][6] += Δb3
    end
end

opt(lr::Float32) = Descent(lr)

function update(Δparamset::Vector, lr::Float32)
    for i in 1:Const.layers_num
        update!(opt(lr), network.f[1][i].W, Δparamset[i][1])
        update!(opt(lr), network.f[1][i].b, Δparamset[i][2])
        update!(opt(lr), network.f[2][i].W, Δparamset[i][3])
        update!(opt(lr), network.f[2][i].b, Δparamset[i][4])
        update!(opt(lr), network.f[3][i].W, Δparamset[i][5])
        update!(opt(lr), network.f[3][i].b, Δparamset[i][6])
    end
end
end
