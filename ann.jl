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

    global o[1]  = Parameters(zeros(Complex{Float32}, Const.dimS, Const.dimB), 
                              zeros(Complex{Float32}, Const.dimS))
    global oe[1] = Parameters(zeros(Complex{Float32}, Const.dimS, Const.dimB), 
                                  zeros(Complex{Float32}, Const.dimS))
    global o[2]  = Parameters(zeros(Complex{Float32}, Const.layer[2], Const.layer[1]), 
                              zeros(Complex{Float32}, Const.layer[2]))
    global oe[2] = Parameters(zeros(Complex{Float32}, Const.layer[2], Const.layer[1]), 
                              zeros(Complex{Float32}, Const.layer[2]))
end

mutable struct Network

    f::Flux.Chain
    p::Zygote.Params
end

function Network()

    func(x::Float32) = tanh(x)
    layer1 = Dense(Const.dimB, Const.dimS)
    layer2 = Dense(Const.layer[1], Const.layer[2])
    f = Chain(layer1, layer2)
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

function init()

    W1 = randn(Float32, Const.dimS, Const.dimB)
    W2 = randn(Float32, Const.layer[2], Const.layer[1])
    b1 = zeros(Float32, Const.dimS)
    b2 = zeros(Float32, Const.layer[2])
    p  = params([W1, b1], [W2, b2])
    Flux.loadparams!(network.f, p)
end

function forward(x::Vector{Float32})

    @views s = x[Const.dimB+1:end]
    @views n = x[1:Const.dimB]
    out = dot(s, network.f[1](n))
    out = network.f[2]([out])

    return out[1] .+ im * out[2]
end

realloss(x::Vector{Float32}) = real(forward(x))
imagloss(x::Vector{Float32}) = imag(forward(x))

function backward(x::Vector{Float32}, e::Complex{Float32})

    realgs = gradient(() -> realloss(x), network.p)
    imaggs = gradient(() -> imagloss(x), network.p)
    for i in 1:Const.layers_num-1
        dw = realgs[network.f[i].W] .- im * imaggs[network.f[i].W]
        db = realgs[network.f[i].b] .- im * imaggs[network.f[i].b]
        o[i].W  += dw
        oe[i].W += dw * e
        o[i].b  += db
        oe[i].b += db * e
    end
    dw = realgs[network.f[end].W] .- im * imaggs[network.f[end].W]
    o[end].W  += dw
    oe[end].W += dw * e
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(energy::Float32, ϵ::Float32, lr::Float32)

    for i in 1:Const.layers_num-1
        ΔW = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
        real.(oe[i].W .- energy * o[i].W) / Const.iters_num
        Δb = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
        real.(oe[i].b .- energy * o[i].b) / Const.iters_num
        update!(opt(lr), network.f[i].W, ΔW)
        update!(opt(lr), network.f[i].b, Δb)
    end
    ΔW = 4.0f0 * (energy - ϵ) * [(energy - ϵ)^2 - Const.η^2 > 0.0f0] .* 
    real.(oe[end].W .- energy * o[end].W) / Const.iters_num
    update!(opt(lr), network.f[end].W, ΔW)
end

end
