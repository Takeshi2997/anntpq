module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, Zygote
using Flux.Optimise: update!
using BSON: @save
using BSON: @load

mutable struct Parameters

    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
end

o  = Vector{Parameters}(undef, Const.layers_num)
oe = Vector{Parameters}(undef, Const.layers_num)

function initO()

    for i in 1:Const.layers_num
        global o[i]  = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
        global oe[i] = Parameters(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                                  zeros(Complex{Float32}, Const.layer[i+1]))
    end
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    func(x::Float32) = x
    W = randn(Complex{Float32}, Const.dimS, Const.dimB)
    b = zeros(Complex{Float32}, Const.dimS)
    layer1  = Dense(W, b)
    f = Chain(layer1)
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

function forward(s::Vector{Float32}, n::Vector{Float32})

    return dot(s, network.f(n))
end

realloss(s::Vector{Float32}, n::Vector{Float32}) = real(forward(s, n))

function backward(s::Vector{Float32}, n::Vector{Float32}, e::Complex{Float32})

    realgs = gradient(() -> realloss(s, n), network.p)
    for i in 1:Const.layers_num
        dw = 2.0 * realgs[network.f[i].W]
        db = 2.0 * realgs[network.f[i].b]
        o[i].W  += dw
        o[i].b  += db
        oe[i].W += dw * e
        oe[i].b += db * e
    end
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)

    for i in 1:Const.layers_num
        ΔW = 2.0f0 * (energy - ϵ) * 2.0f0 * 
        real.(oe[i].W .- energy * o[i].W) / Const.iters_num
        Δb = 2.0f0 * (energy - ϵ) * 2.0f0 * 
        real.(oe[i].b .- energy * o[i].b) / Const.iters_num
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
end

end
